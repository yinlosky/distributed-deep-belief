package edu.uci.ics.DDBN;

import java.io.*;
import java.util.*;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

import org.jblas.DoubleMatrix;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.util.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.LineRecordReader;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class BatchGenerationEngine extends Configured implements Tool {
	public static class BatchInputFormat extends FileInputFormat<IntWritable,Text> {

		@Override
		public RecordReader<IntWritable, Text> createRecordReader(InputSplit is,
				TaskAttemptContext tac) throws IOException,
				InterruptedException {
			ImageReader ir = new ImageReader();
			ir.initialize(is,tac);
			return ir;
		}
		
	}
	public static class ImageReader extends RecordReader<IntWritable,Text> {
		private LineRecordReader lineReader;
		private IntWritable lineKey;
		private Text lineValue;		
		
		@Override
		public void close() throws IOException {
			lineReader.close();
		}

		@Override
		public IntWritable getCurrentKey() throws IOException, InterruptedException {
			return lineKey;
		}

		@Override
		public Text getCurrentValue() throws IOException, InterruptedException {
			return lineValue;
		}

		@Override
		public float getProgress() throws IOException, InterruptedException {
			return lineReader.getProgress();
		}

		@Override
		public void initialize(InputSplit is, TaskAttemptContext tac)
				throws IOException, InterruptedException {
			lineReader = new LineRecordReader();
			lineReader.initialize(is, tac);
		}

		@Override
		public boolean nextKeyValue() throws IOException, InterruptedException {
			if(!lineReader.nextKeyValue()) {
				return false;
			}
			String[] segment = lineReader.getCurrentValue().toString().split("\t");
			lineKey = new IntWritable(Integer.parseInt(segment[0]));
			lineValue = new Text(segment[1] + "\t" + segment[2]);
			return true;
		}
		
	}
	public static class ImageSplit extends Mapper<IntWritable,Text,IntWritable,Text> {
		private IntWritable sameKey = new IntWritable(0);
		private Text dataString = new Text();
		
		public void map(IntWritable key, Text value,
				OutputCollector<IntWritable,Text> output, Reporter reporter) throws IOException {
			String dataRow = value.toString();
			StringTokenizer tk = new StringTokenizer(dataRow);
			String label = tk.nextToken();
			String image = tk.nextToken();
			dataString.set(label + "\t" + image);			
			output.collect(sameKey, dataString);
		}
	}
	public static class Minibatcher extends Reducer<IntWritable,Text,Text,jBLASArrayWritable> {
		private Text batchID = new Text();
		private jBLASArrayWritable dataArray;
		
		private int visibleNodes = 28*28;
		private int hiddenNodes = 500;
		private int exampleCount = 60000;
		private int batchSize = 20;
		
		public void configure(Configuration conf) {
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
						if(elName == "visible.nodes") {
							this.visibleNodes = Integer.parseInt(xmlGetSingleValue(property,"value"));
						} else if(elName == "hidden.nodes") {
							this.hiddenNodes = Integer.parseInt(xmlGetSingleValue(property,"value"));
						} else if(elName == "example.count") {
							this.exampleCount = Integer.parseInt(xmlGetSingleValue(property,"value"));
						} else if(elName == "batch.size") {
							this.batchSize = Integer.parseInt(xmlGetSingleValue(property,"value"));
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
		
		public void reduce(IntWritable sameNum, Iterator<Text> data,
				OutputCollector<Text,jBLASArrayWritable> output, Reporter reporter) throws IOException {
			int totalBatchCount = exampleCount/batchSize;
			
			DoubleMatrix weights = DoubleMatrix.randn(hiddenNodes,visibleNodes);
			DoubleMatrix hbias = DoubleMatrix.zeros(hiddenNodes);
			DoubleMatrix vbias = DoubleMatrix.zeros(visibleNodes);
			DoubleMatrix label = DoubleMatrix.zeros(1);
			DoubleMatrix hidden_chain = null;
			DoubleMatrix vdata = DoubleMatrix.zeros(batchSize,visibleNodes);
			
			ArrayList<DoubleMatrix> outputmatricies = new ArrayList<DoubleMatrix>();
			outputmatricies.add(weights);
			outputmatricies.add(hbias);
			outputmatricies.add(vbias);
			outputmatricies.add(label);
			outputmatricies.add(vdata);
			outputmatricies.add(hidden_chain);
			
			int j;
			for(int i = 0; i < totalBatchCount; i++) {
				j = 0;
				while(data.hasNext() && j < batchSize ) {
					j++;
					StringTokenizer tk = new StringTokenizer(data.next().toString());
					label.put(0, Double.parseDouble(tk.nextToken()));
					String image = tk.nextToken();
					for(int k = 0; k < image.length(); k++) {
						Integer val = new Integer( image.charAt(k) );
						vdata.put(j,k, val.doubleValue());
					}
					dataArray = new jBLASArrayWritable(outputmatricies);
					batchID.set("1\t"+i);
					output.collect(batchID, dataArray);
				}
			}
		}	
	}
	
	@Override
	public int run(String[] args) throws Exception {
		Configuration conf = getConf();
		
		Job job = new Job(conf,"minibatch");
		
		job.setJarByClass(BatchGenerationEngine.class);
		
		job.setInputFormatClass(BatchInputFormat.class);
		
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(jBLASArrayWritable.class);
		
		job.setMapperClass(ImageSplit.class);
		job.setCombinerClass(Minibatcher.class);
		job.setReducerClass(Minibatcher.class);
				
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
				conf.setBoolean("minibatch.job.setup",true);
			} else {
				other_args.add(inputArgs[i]);
			}
		}
		
		String[] tool_args = other_args.toArray(new String[0]);
		int result = ToolRunner.run(conf, new BatchGenerationEngine(), tool_args);
		System.exit(result);
	}	
}
