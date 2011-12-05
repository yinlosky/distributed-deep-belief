package edu.uci.ics.DDBN;


import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.jBLASArrayWritable;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileRecordReader;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.*;
import org.jblas.DoubleMatrix;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

public class BinarySigmoidRBM extends Configured implements Tool {
	public static class BatchFormat extends FileInputFormat<Text,jBLASArrayWritable> {

		@Override
		public RecordReader<Text, jBLASArrayWritable> createRecordReader(
				InputSplit split, TaskAttemptContext context)
				throws IOException, InterruptedException {
			BatchReader br = new BatchReader();
			br.initialize(split,context);
			return br;
		}
		
	}
	
	public static class BatchReader extends RecordReader<Text,jBLASArrayWritable> {
		private SequenceFileRecordReader<Text, jBLASArrayWritable> reader;
		Text key;
		jBLASArrayWritable value;
		
		@Override
		public void close() throws IOException {
			reader.close();
		}

		@Override
		public Text getCurrentKey() throws IOException, InterruptedException {
			return key;
		}

		@Override
		public jBLASArrayWritable getCurrentValue() throws IOException,
				InterruptedException {
			return value;
		}

		@Override
		public float getProgress() throws IOException, InterruptedException {
			return reader.getProgress();
		}

		@Override
		public void initialize(InputSplit split, TaskAttemptContext context)
				throws IOException, InterruptedException {
			reader = new SequenceFileRecordReader<Text,jBLASArrayWritable>();
			key = new Text();
			value = new jBLASArrayWritable();
			reader.initialize(split, context);
		}

		@Override
		public boolean nextKeyValue() throws IOException, InterruptedException {
			if(!reader.nextKeyValue()) {
				return false;
			}
			key = new Text();
			value = new jBLASArrayWritable();			
			
			key = reader.getCurrentKey();
			value = reader.getCurrentValue();
			return true;
		}
		
	}
	
	public static class GibbsSampling extends Mapper<Text, jBLASArrayWritable, Text, jBLASArrayWritable> {
		private DoubleMatrix weights,hbias,hiddenChain;
		private DoubleMatrix vbias,label,v_data;
		private DoubleMatrix h1_data;
		private DoubleMatrix v1_data;
		private DoubleMatrix w1,hb1;
		private DoubleMatrix vb1;
		
		private int gibbsSteps = 1;
		private double learningRate = 0.1;
		private int layers = 1;
		
		@Override
		public void setup(GibbsSampling.Context context) {
			Configuration conf = context.getConfiguration();
			if (conf.getBoolean("minibatch.job.setup", false)) {
				Path[] jobSetupFiles = new Path[0];
				try {
					jobSetupFiles = DistributedCache.getLocalCacheFiles(conf);	
				} catch (IOException ioe) {
					System.err.println("Caught exception while getting cached files: " + StringUtils.stringifyException(ioe));
				}
				for (Path jobSetup : jobSetupFiles) {
					parseJobSetup(jobSetup);
				}
			}
		}
		
		private String xmlGetSingleValue(Element el, String tag) {
			return ((Element)el.getElementsByTagName(tag).item(0)).getFirstChild().getNodeValue();
		}
		
		private void parseJobSetup(Path jobFile) {
			DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
			try {
				DocumentBuilder db = dbf.newDocumentBuilder();
				Document doc = db.parse(jobFile.toString());
				Element configElement = doc.getDocumentElement();
				NodeList nodes = configElement.getElementsByTagName("property");
				if(nodes != null && nodes.getLength() > 0) {
					for(int i = 0; i < nodes.getLength(); i++) {
						Element property = (Element)nodes.item(i);
						String elName = xmlGetSingleValue(property,"name");
						if(elName.compareToIgnoreCase("gibbs.steps") == 0) {
							this.gibbsSteps = Integer.parseInt(xmlGetSingleValue(property,"value"));
						} else if(elName.compareToIgnoreCase("learning.rate") == 0) {
							this.learningRate = Double.parseDouble(xmlGetSingleValue(property,"value"));
						} else if(elName.compareToIgnoreCase("layer.count") == 0) {
							this.layers = Integer.parseInt(xmlGetSingleValue(property,"value"));
						}
					}
				}
				
			} catch (ParserConfigurationException pce) {
				System.err.println("Caught exception while parsing the cached file '" + jobFile + "' : " + StringUtils.stringifyException(pce));
			} catch(SAXException se) {
				System.err.println("Caught exception while parsing the cached file '" + jobFile + "' : " + StringUtils.stringifyException(se));
			}catch(IOException ioe) {
				System.err.println("Caught exception while parsing the cached file '" + jobFile + "' : " + StringUtils.stringifyException(ioe));
			}
		}
		//TODO: DOUBLECHECK EVERYTHING
		@Override
		public void map(Text key,
				jBLASArrayWritable input,
				GibbsSampling.Context context) throws IOException, InterruptedException {
			/* *******************************************************************/
			/* initialize all memory we're going to use during the process		 */
			
			ArrayList<DoubleMatrix> data = input.getData();
			weights = data.get(0);
			hbias = data.get(1);
			hiddenChain = data.get(2);
			vbias = data.get(3);
			label = data.get(4);
			v_data = data.get(5);
			
			w1 = DoubleMatrix.zeros(weights.rows,weights.columns);
			hb1 = DoubleMatrix.zeros(hbias.rows,hbias.columns);
			vb1 = DoubleMatrix.zeros(vbias.rows,vbias.columns);
			
			DoubleMatrix mean_positive_phase = new DoubleMatrix(v_data.rows,hbias.columns);
						
			/* ********************************************************************/
			// sample hidden state to get positive phase
			// if empty, use it as the start of the chain
			// or use persistent hidden state from pCD		
			
			DoubleMatrix phaseSample = sample_h_from_v(v_data,mean_positive_phase);
			
			if(hiddenChain == null) {
				hiddenChain = new DoubleMatrix(hbias.rows,hbias.columns);
				h1_data.copy(phaseSample);
			}
			else {
				h1_data.copy(hiddenChain);
			}				
			// run Gibbs chain for k steps
			for(int j = 0; j < gibbsSteps; j++) {
				v1_data.copy(sample_v_from_h(h1_data));
				h1_data.copy(sample_h_from_v(v1_data));
			}
			weight_contribution(hiddenChain,v_data,	h1_data,v1_data);
			hiddenChain.copy(h1_data);
			
			data.get(0).copy(w1);
			data.get(1).copy(hb1);
			data.get(2).copy(hiddenChain);
			data.get(3).copy(vb1);
			
			jBLASArrayWritable outputmatrix = new jBLASArrayWritable(data);
			context.write(key, outputmatrix);
		}
		
		public DoubleMatrix sample_h_from_v(DoubleMatrix v0) {
			return matrixBinomial(propup(v0));
		}
		
		public DoubleMatrix sample_h_from_v(DoubleMatrix v0, DoubleMatrix phase) {
			DoubleMatrix activations = propup(v0);
			phase.copy(activations);
			return matrixBinomial(activations);
		}
		
		public DoubleMatrix propup(DoubleMatrix v) {
			return sigmoid(this.weights.mul(v.transpose()).transpose()
						.add(hbias.repmat(v_data.rows, 1)));
		}
		
		public DoubleMatrix sample_v_from_h(DoubleMatrix h0) {
			return matrixBinomial(propdown(h0));
		}
		
		public DoubleMatrix propdown(DoubleMatrix h) {
			return sigmoid(h.mul(weights).add(vbias.repmat(20,1)));
		}
				
		public DoubleMatrix sigmoid(DoubleMatrix exponent) {
			
			int rows = exponent.rows, cols = exponent.columns;
			double[][] p_y_given_x = new double[rows][cols];
			for(int i = 0; i < rows; i++) {
				for (int j = 0; j < cols; j++) {
					p_y_given_x[i][j] = 1.0 / (1.0 + Math.exp(exponent.get(i, j)));
				}
			}			
			return new DoubleMatrix(p_y_given_x);
		}
		
		public DoubleMatrix matrixBinomial(DoubleMatrix p) {
			int rows = p.rows, cols = p.columns;
			double[][] result = new double[rows][cols];
			for(int i = 0; i < rows; i++) {
				for (int j = 0; j < cols; j++) {
					result[i][j] = Math.random() < p.get(i,j) ? 1.0 : 0.0;
				}
			}	
			return new DoubleMatrix(result);
		}
						
		public void weight_contribution(DoubleMatrix h0,DoubleMatrix v0,
				DoubleMatrix h1,DoubleMatrix v1) {
			this.w1.add(h0.transpose().mul(v0).sub(h1.transpose().mul(v1)).mul(learningRate));
			this.vb1.add(v0.sub(v1).mul(learningRate)).columnMeans();
			this.hb1.add(h0.sub(h1).mul(learningRate)).columnMeans();
		}
	}
	
	//TODO: FIX REDUCER
	public static class WeightContributions extends Reducer<Text, jBLASArrayWritable, Text, jBLASArrayWritable> {

		@Override
		public void reduce(Text key,
				Iterable<jBLASArrayWritable> inputs,
				WeightContributions.Context context) throws IOException, InterruptedException {
			DoubleMatrix weights = null,hbias = null,vbias = null;
			
			ArrayList<DoubleMatrix> output_array = new ArrayList<DoubleMatrix>();
			output_array.add(weights);
			output_array.add(hbias);
			output_array.add(vbias);
			
			for(jBLASArrayWritable input : inputs) {
				ArrayList<DoubleMatrix> data = input.getData();
				DoubleMatrix w_cont = data.get(0);
				DoubleMatrix hb_cont = data.get(1);
				DoubleMatrix vb_cont = data.get(2);
				
				if(weights == null) {
					weights = DoubleMatrix.zeros(w_cont.rows,w_cont.columns);
					hbias = DoubleMatrix.zeros(hb_cont.rows,hb_cont.columns);
					vbias = DoubleMatrix.zeros(vb_cont.rows,vb_cont.columns);
				}
				weights.add(w_cont);
				hbias.add(hb_cont);
				vbias.add(vb_cont);
				output_array.add(data.get(3));
			}
			jBLASArrayWritable outputmatrix = new jBLASArrayWritable(output_array);
			context.write(key,outputmatrix);
		}
		
	}
	
	@Override
	public int run(String[] args) throws Exception {
		Configuration conf = getConf();
		
		Job job = new Job(conf,"rbmstep");
		
		job.setJarByClass(BinarySigmoidRBM.class);
		
		job.setInputFormatClass(BatchFormat.class);
		
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(jBLASArrayWritable.class);
		
		job.setMapperClass(GibbsSampling.class);
		job.setReducerClass(WeightContributions.class);
		
		job.setNumReduceTasks(1);
				
		FileInputFormat.setInputPaths(job, new Path(args[0]));
		FileOutputFormat.setOutputPath(job, new Path(args[1]));
		
		return job.waitForCompletion(true) ? 0 : 1;
	}
	
	public static void main(String[] args) throws Exception{
		System.out.println("Beginning job...");
		Configuration conf = new Configuration();
		String[] inputArgs = new GenericOptionsParser(conf,args).getRemainingArgs();
		
		List<String> other_args = new ArrayList<String>();
		for(int i = 0; i < args.length; ++i) {
			if("-setup".equals(inputArgs[i])) {
				DistributedCache.addCacheFile(new Path(inputArgs[++i]).toUri(),conf);
				conf.setBoolean("rbmstep.job.setup",true);
			} else {
				other_args.add(inputArgs[i]);
			}
		}
		
		String[] tool_args = other_args.toArray(new String[0]);
		int result = ToolRunner.run(conf, new BinarySigmoidRBM(), tool_args);
		System.exit(result);
	}
}
