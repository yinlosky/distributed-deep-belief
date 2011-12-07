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
import org.apache.log4j.Logger;
import org.jblas.DoubleMatrix;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

import edu.uci.ics.DDBN.BatchGenerationEngine.BatchOutputFormat;

public class BinarySigmoidRBM extends Configured implements Tool {
	static Logger log = Logger.getLogger(BinarySigmoidRBM.class);
	
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
		private double weightCost = 0.0002;
		private int classCount = 10;
		
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
						} else if(elName.compareToIgnoreCase("weight.cost") == 0) {
							this.weightCost = Double.parseDouble(xmlGetSingleValue(property,"value"));
						} else if(elName.compareToIgnoreCase("class.count") == 0) {
							this.classCount = Integer.parseInt(xmlGetSingleValue(property,"value"));
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
			long start_time = System.nanoTime();
			ArrayList<DoubleMatrix> data = input.getData();
			label = data.get(4);
			v_data = data.get(5);
			
			//check to see if we are in the first layer or there are layers beneath us we must sample from
			if(data.size() > 6) {
				int prelayer = (data.size()-6)/4;			
				DoubleMatrix[] preWeights = new DoubleMatrix[prelayer],
					preHbias = new DoubleMatrix[prelayer],
					preVbias = new DoubleMatrix[prelayer];
				for(int i = 0; i < prelayer; i++ ) {
					preWeights[i] = data.get(6+i*3);
					preHbias[i] = data.get(7+i*3);
					preVbias[i] = data.get(8+i*3);
				}
				DoubleMatrix vnew = null;
				for(int i = 0; i < prelayer; i++) {
					weights = preWeights[i];
					vbias = preVbias[i];
					hbias = preHbias[i];
					vnew = sample_h_from_v(i==0 ? v_data : vnew);
				}
				v_data = vnew;
			}
			
			weights = data.get(0);
			hbias = data.get(1);
			hiddenChain = data.get(2);
			vbias = data.get(3);
			
			//check if we need to attach labels to the observed variables
			if(vbias.columns != v_data.columns) {
				DoubleMatrix labels = DoubleMatrix.zeros(1,classCount);
				int labelNum = (new Double(label.get(0))).intValue();
				labels.put(labelNum,1.0);
				v_data = DoubleMatrix.concatHorizontally(v_data, labels);
			}	
			
			w1 = DoubleMatrix.zeros(weights.rows,weights.columns);
			hb1 = DoubleMatrix.zeros(hbias.rows,hbias.columns);
			vb1 = DoubleMatrix.zeros(vbias.rows,vbias.columns);
						
			/* ********************************************************************/
			// sample hidden state to get positive phase
			// if empty, use it as the start of the chain
			// or use persistent hidden state from pCD		
			
			DoubleMatrix phaseSample = sample_h_from_v(v_data);
			h1_data = new DoubleMatrix();			
			v1_data = new DoubleMatrix();
			
			if(hiddenChain == null) {
				data.set(2, new DoubleMatrix(hbias.rows,hbias.columns));
				hiddenChain = data.get(2);
				hiddenChain.copy(phaseSample);
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
			DoubleMatrix hprob = propup(v1_data);
			weight_contribution(hiddenChain,v_data,	hprob,v1_data);
			hiddenChain.copy(h1_data);
			
			data.get(0).copy(w1);
			data.get(1).copy(hb1);
			data.get(2).copy(hiddenChain);
			data.get(3).copy(vb1);
			
			jBLASArrayWritable outputmatrix = new jBLASArrayWritable(data);
			context.write(key, outputmatrix);
			log.info("Job completed in: " + (System.nanoTime()-start_time)/(1E6) + " ms");
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
			return sigmoid(this.weights.mmul(v.transpose()).transpose()
						.addi(hbias.repmat(v_data.rows, 1)));
		}
		
		public DoubleMatrix sample_v_from_h(DoubleMatrix h0) {
			return matrixBinomial(propdown(h0));
		}
		
		public DoubleMatrix propdown(DoubleMatrix h) {
			return sigmoid(h.mmul(this.weights)
					.addi(vbias.repmat(v_data.rows,1)));
		}
				
		public DoubleMatrix sigmoid(DoubleMatrix exponent) {
			
			int rows = exponent.rows, cols = exponent.columns;
			for(int i = 0; i < rows; i++) {
				for (int j = 0; j < cols; j++) {
					exponent.put(i,j, 1.0 / (1.0 + Math.exp(exponent.get(i, j)*-1.0)));
				}
			}			
			return exponent;
		}
		
		public DoubleMatrix matrixExponential(DoubleMatrix exponent) {
			int rows = exponent.rows, cols = exponent.columns;
			for(int i = 0; i < rows; i++) {
				for(int j = 0; j < cols; j++) {
					exponent.put(i,j, Math.exp(exponent.get(i,j)));
				}
			}
			return exponent;
		}
		
		public DoubleMatrix matrixLogarithm(DoubleMatrix logponent) {
			int rows = logponent.rows, cols = logponent.columns;
			for(int i = 0; i < rows; i++) {
				for(int j = 0; j < cols; j++) {
					logponent.put(i,j, Math.log(logponent.get(i,j)));
				}
			}
			return logponent;
		}
		
		public DoubleMatrix matrixBinomial(DoubleMatrix p) {
			int rows = p.rows, cols = p.columns;
			for(int i = 0; i < rows; i++) {
				for (int j = 0; j < cols; j++) {
					p.put(i,j, Math.random() < p.get(i,j) ? 1.0 : 0.0);
				}
			}	
			return p;
		}
		
		public DoubleMatrix matrixSum(DoubleMatrix summant, int axis ) {
			int rows = axis==0 ? summant.rows : 1, cols = axis==0 ? 1 : summant.columns;
			DoubleMatrix sum = new DoubleMatrix(rows,cols);
			if(axis == 0) {
				for(int i = 0; i < summant.columns; i++) {
					sum.put(1,i, summant.getColumn(i).sum());
				}
			} else {
				for(int i = 0; i < summant.rows; i++) {
					sum.put(i,1, summant.getRow(i).sum());
				}
			}
			return sum;
		}
						
		public void weight_contribution(DoubleMatrix h0,DoubleMatrix v0,
				DoubleMatrix h1,DoubleMatrix v1) {
			this.w1.addi(h0.transpose().mmul(v0)
					.subi(h1.transpose().mmul(v1)))
					.muli(learningRate/v_data.rows).subi(weights.mul(weightCost));
			
			this.vb1.addi(v0.sub(v1).muli(learningRate/v_data.rows).columnMeans());
			this.hb1.addi(h0.sub(h1).muli(learningRate/v_data.rows).columnMeans());
		}
		
		public DoubleMatrix free_energy(DoubleMatrix v_sample) {
			DoubleMatrix wv_hb = weights.mmul(v_sample.transpose()).addi(this.hbias.repmat(v_sample.rows,1).transpose());
			DoubleMatrix vb = v_sample.mmul(this.vbias.transpose());
			DoubleMatrix hi = matrixSum(matrixLogarithm(matrixExponential(wv_hb).addi(1.0)),1);
			return hi.mul(-1.0).subi(vb);
		}
	}
	
	//TODO: FIX REDUCER
	public static class WeightContributions extends Reducer<Text, jBLASArrayWritable, Text, jBLASArrayWritable> {

		@Override
		public void reduce(Text key,
				Iterable<jBLASArrayWritable> inputs,
				WeightContributions.Context context) throws IOException, InterruptedException {
			DoubleMatrix w_cont = new DoubleMatrix(),
				hb_cont = new DoubleMatrix(),
				vb_cont = new DoubleMatrix(),
				weights = null,
				hbias = null,
				vbias = null;
			
			ArrayList<DoubleMatrix> chainList = new ArrayList<DoubleMatrix>();
			
			ArrayList<DoubleMatrix> output_array = new ArrayList<DoubleMatrix>();
			
			int count = 0;
			
			for(jBLASArrayWritable input : inputs) {
				ArrayList<DoubleMatrix> data = input.getData();
				w_cont.copy(data.get(0));
				hb_cont.copy(data.get(1));
				vb_cont.copy(data.get(3));
				
				//save list of all hidden chains for updates to batch files in phase 3
				chainList.add(new DoubleMatrix(data.get(2).toArray2()));
				
				if(weights == null) {
					weights = DoubleMatrix.zeros(w_cont.rows,w_cont.columns);
					hbias = DoubleMatrix.zeros(hb_cont.rows,hb_cont.columns);
					vbias = DoubleMatrix.zeros(vb_cont.rows,vb_cont.columns);
				}
				
				//sum weight contributions
				weights.addi(w_cont);
				hbias.addi(hb_cont);
				vbias.addi(vb_cont);
				count++;
			}
			
			output_array.add(weights.div(count));
			output_array.add(hbias.div(count));
			output_array.add(vbias.div(count));
			output_array.addAll(chainList);			
			
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
		
		job.setOutputFormatClass(BatchOutputFormat.class);
				
		FileInputFormat.setInputPaths(job, new Path(args[0]));
		FileOutputFormat.setOutputPath(job, new Path(args[1]));
		
		return job.waitForCompletion(true) ? 0 : 1;
	}
	
	public static void main(String[] args) throws Exception{
		log.info("Beginning job...");
		Configuration conf = new Configuration();
		String[] inputArgs = new GenericOptionsParser(conf,args).getRemainingArgs();
		
		List<String> other_args = new ArrayList<String>();
		for(int i = 0; i < args.length; ++i) {
			if("-setup".equals(inputArgs[i])) {
				DistributedCache.addCacheFile(new Path(inputArgs[++i]).toUri(),conf);
				conf.setBoolean("minibatch.job.setup",true);
			} else {
				other_args.add(inputArgs[i]);
			}
		}
		
		String[] tool_args = other_args.toArray(new String[0]);
		int result = ToolRunner.run(conf, new BinarySigmoidRBM(), tool_args);
		System.exit(result);
	}
}
