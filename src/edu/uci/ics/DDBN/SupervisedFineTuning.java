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
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LabelImageWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.jBLASArrayWritable;
import org.apache.hadoop.io.SequenceFile.Reader;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.hadoop.util.StringUtils;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;
import org.jblas.DoubleMatrix;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

import edu.uci.ics.DDBN.BatchGenerationEngine.BatchInputFormat;
import edu.uci.ics.DDBN.BatchGenerationEngine.BatchOutputFormat;
import edu.uci.ics.DDBN.BinarySigmoidRBM.BatchFormat;

public class SupervisedFineTuning extends Configured implements Tool {
	static Logger log = Logger.getLogger(SupervisedFineTuning.class);
	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		System.out.println("Beginning job...");
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
		int result = ToolRunner.run(conf, new SupervisedFineTuning(), tool_args);
		System.exit(result);
	}

	@Override
	public int run(String[] args) throws Exception {
		Configuration conf = getConf();
		
		Job job = new Job(conf,"rbmstep");
		
		job.setJarByClass(BinarySigmoidRBM.class);
		
		job.setInputFormatClass(BatchInputFormat.class);
		
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(jBLASArrayWritable.class);
		
		job.setMapperClass(GradientDescent.class);
		job.setReducerClass(SumGradients.class);
		
		job.setNumReduceTasks(1);
		
		job.setOutputFormatClass(BatchOutputFormat.class);
				
		FileInputFormat.setInputPaths(job, new Path(args[0]));
		FileOutputFormat.setOutputPath(job, new Path(args[1]));
		
		return job.waitForCompletion(true) ? 0 : 1;
	}
	
	public static class GradientDescent extends Mapper<Text,LabelImageWritable,Text,jBLASArrayWritable> {
		private int layerCount = 3;
		private double learningRate = 0.1;
		private double weightCost = 0.0002;
		private int classCount = 10;
		private ArrayList<DoubleMatrix> weightData;
				
		@Override
		public void setup(GradientDescent.Context context) {
			Configuration conf = context.getConfiguration();
			if (conf.getBoolean("minibatch.job.setup", false)) {
				Path[] jobSetupFiles = new Path[0];
				try {
					jobSetupFiles = DistributedCache.getLocalCacheFiles(conf);	
				} catch (IOException ioe) {
					System.err.println("Caught exception while getting cached files: " + StringUtils.stringifyException(ioe));
				}
				for (Path jobSetup : jobSetupFiles) {
					parseJobSetup(jobSetup, context);
				}
			}
		}
		
		@Override
		public void map(Text key,
				LabelImageWritable value,
				GradientDescent.Context context) throws IOException {
			byte[] data = value.getImage();
			double label = (new Integer(value.getLabel())).doubleValue();
			DoubleMatrix v_data = new DoubleMatrix(1,data.length);
			
			for(int i = 0; i < data.length; i++) {
				v_data.put(0,i, (int)data[i]);
			}

			DoubleMatrix[] weights = new DoubleMatrix[layerCount],
				hbias = new DoubleMatrix[layerCount+1];
			
			int ptr = 1;
			
			int epoch = (new Double(weightData.get(0).get(0))).intValue();
			
			for(int i = 0; i < layerCount; i++) {
				weights[i] = weightData.get(i+ptr);
			}
			ptr += layerCount;
			for(int i = 0; i < layerCount+1; i++) {
				hbias[i] = weightData.get(i+ptr);
			}
			ptr += layerCount+1;
			DoubleMatrix Vweight = weightData.get(ptr);
			DoubleMatrix Vbias = weightData.get(ptr+1);
						
			DoubleMatrix[] meanField = new DoubleMatrix[layerCount];
			meanField[0] = v_data;
			
			for(int i = 1; i <= layerCount; i++) {
				meanField[i] = meanSample(weights[i-1],
						meanField[i-1],
						hbias[i]);				
			}
			
			
						
			if(key.toString().equalsIgnoreCase("test")) {
				DoubleMatrix targetout = MatrixMath.exp(Vweight.mmul(meanField[layerCount]));
				targetout = targetout.diviRowVector(MatrixMath.sum(targetout,0));
				
			} else {
				
			}
		}
		
		public DoubleMatrix meanSample(DoubleMatrix layerW,
				DoubleMatrix layerMu,
				DoubleMatrix layerBias) {
			return MatrixMath.sigmoid(layerMu.mmul(layerW)
					.add(layerBias.repmat(layerMu.rows,1)));
		}
		
		private static String xmlGetSingleValue(Element el, String tag) {
			return ((Element)el.getElementsByTagName(tag).item(0))
						.getFirstChild().getNodeValue();
		}
		
		private void parseJobSetup(Path jobFile, GradientDescent.Context context) {
			DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
			
			if(jobFile.getName().endsWith("xml")) {
				try {
					DocumentBuilder db = dbf.newDocumentBuilder();
					Document doc = db.parse(jobFile.toString());
					Element configElement = doc.getDocumentElement();
					NodeList nodes = configElement.getElementsByTagName("property");
					if(nodes != null && nodes.getLength() > 0) {
						for(int i = 0; i < nodes.getLength(); i++) {
							Element property = (Element)nodes.item(i);
							String elName = xmlGetSingleValue(property,"name");
							if(elName.compareToIgnoreCase("layer.count") == 0) {
								this.layerCount = Integer.parseInt(xmlGetSingleValue(property,"value"));
							} else if(elName.compareToIgnoreCase("supervised.learning.rate") == 0) {
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
			} else {
				Configuration conf = context.getConfiguration();
				SequenceFile.Reader reader = null;
				try{
					FileSystem fs = FileSystem.get(conf);
					reader = new SequenceFile.Reader(fs, jobFile, conf);
				} catch(IOException ioe) {
					try{
						FileSystem fs = FileSystem.getLocal(conf);
						reader = new SequenceFile.Reader(fs,jobFile,conf);
					} catch(IOException ioe2) {
						log.fatal(ioe2);
						throw new RuntimeException();
					}					
				}
				try{
					Text cachekey = new Text();
					jBLASArrayWritable cachevalue = new jBLASArrayWritable();
					reader.next(cachekey,cachevalue);
					weightData = cachevalue.getData();
				} catch(IOException ioe) {
					log.fatal(ioe);
					throw new RuntimeException();
				}
			}
		}
	}
	
	public static class SumGradients extends Reducer<Text,jBLASArrayWritable,Text,jBLASArrayWritable> {
		@Override
		public void reduce(Text key,
				Iterable<jBLASArrayWritable> data,
				SumGradients.Context context) throws IOException {
			
		}
	}
}
