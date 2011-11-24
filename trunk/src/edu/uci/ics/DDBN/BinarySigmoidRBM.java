package edu.uci.ics.DDBN;


import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.jBLASArrayWritable;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.*;
import org.jblas.DoubleMatrix;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;


public class BinarySigmoidRBM extends Configured implements Tool {
	public static class GibbsSampling extends Mapper<LongWritable, jBLASArrayWritable, LongWritable, jBLASArrayWritable> {
		private DoubleMatrix weights,hbias,vbias;
		private List<DoubleMatrix> hidden_state,v_data;
		private DoubleMatrix v1_data,h1_data;
		private DoubleMatrix w1,hb1,vb1;
		
		private int gibbsSteps = 1;
		private double learningRate = 0.1;
		
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
						if(elName == "gibbs.steps") {
							this.gibbsSteps = Integer.parseInt(xmlGetSingleValue(property,"value"));
						} else if(elName == "learning.rate") {
							this.learningRate = Double.parseDouble(xmlGetSingleValue(property,"value"));
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
		
		public void map(LongWritable key,
				jBLASArrayWritable input,
				OutputCollector<LongWritable, jBLASArrayWritable> output,
				Reporter reporter) throws IOException {
			ArrayList<DoubleMatrix> data = input.getData();
			weights = data.get(0);
			hbias = data.get(1);
			vbias = data.get(2);
			
			w1 = DoubleMatrix.zeros(weights.rows,weights.columns);
			hb1 = DoubleMatrix.zeros(hbias.rows,hbias.columns);
			vb1 = DoubleMatrix.zeros(vbias.rows,vbias.columns);
			
			int N = (data.size()-3)/2;
			
			hidden_state = data.subList(3,3+N);
			v_data = data.subList(3+N,3+2*N);
			
			for(int i = 0; i < N; i++) {
				//sample hidden state if empty or use current hidden state
								
				if(hidden_state == null) {
					hidden_state = new ArrayList<DoubleMatrix>(v_data.size());
					h1_data = sample_h_from_v(v_data.get(i));
				}
				else {
					h1_data = hidden_state.get(i);
				}
				for(int j = 0; j < gibbsSteps; j++) {
					v1_data = sample_v_from_h(h1_data);
					h1_data = sample_h_from_v(v1_data);
				}
				hidden_state.set(i, h1_data);
				weight_contribution(hidden_state.get(i),v_data.get(i),h1_data,v1_data);
			}
			data.set(0, w1);
			data.set(1,hb1);
			data.set(2,vb1);
			for(int i = 0; i < hidden_state.size(); i++) {
				data.set(3+i,hidden_state.get(i));
			}
			jBLASArrayWritable outputmatrix = new jBLASArrayWritable(data);
			output.collect(key, outputmatrix);
		}
		
		public DoubleMatrix sample_h_from_v(DoubleMatrix v0) {
			return matrixBinomial(propup(v0));
		}
		
		public DoubleMatrix propup(DoubleMatrix v) {
			return sigmoid(v.mul(weights).add(hbias));
		}
		
		public DoubleMatrix sample_v_from_h(DoubleMatrix h0) {
			return matrixBinomial(propdown(h0));
		}
		
		public DoubleMatrix propdown(DoubleMatrix h) {
			return sigmoid(h.mul(weights.transpose()).add(vbias));
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
						
		public void weight_contribution(DoubleMatrix h0,DoubleMatrix v0,DoubleMatrix h1,DoubleMatrix v1) {
			this.w1.add(v0.transpose().mul(h0).sub(v1.transpose().mul(h1)).mul(learningRate));
			this.vb1.add(v0.sub(v1).mul(learningRate));
			this.hb1.add(h0.sub(h1).mul(learningRate));
		}
	}
	public static class WeightContributions extends Reducer<LongWritable, jBLASArrayWritable, LongWritable, jBLASArrayWritable> {

		public void reduce(LongWritable key,
				Iterator<jBLASArrayWritable> inputs,
				OutputCollector<LongWritable, jBLASArrayWritable> output,
				Reporter reporter) throws IOException {
			DoubleMatrix weights = null,hbias = null,vbias = null;
			
			ArrayList<DoubleMatrix> output_array = new ArrayList<DoubleMatrix>();
			output_array.add(weights);
			output_array.add(hbias);
			output_array.add(vbias);
			
			while(inputs.hasNext()) {
				jBLASArrayWritable input = inputs.next();
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
			output.collect(key,outputmatrix);
		}
		
	}
	
	@Override
	public int run(String[] args) throws Exception {
		Configuration conf = getConf();
		
		Job job = new Job(conf,"rbmstep");
		
		job.setJarByClass(BinarySigmoidRBM.class);
		
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(jBLASArrayWritable.class);
		
		job.setMapperClass(GibbsSampling.class);
		job.setCombinerClass(WeightContributions.class);
		job.setReducerClass(WeightContributions.class);
				
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
