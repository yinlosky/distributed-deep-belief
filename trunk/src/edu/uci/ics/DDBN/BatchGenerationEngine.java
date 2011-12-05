//bin/hadoop jar minibatch.jar edu.uci.ics.DDBN.BatchGenerationEngine -setup /user/hadoop/batch_conf/batch_conf.xml

package edu.uci.ics.DDBN;

//base java libraries
import java.io.*;
import java.util.*;

//xml libraries
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;
//exterior libraries
import org.apache.log4j.Logger;
import org.jblas.DoubleMatrix;
//hadoop libraries
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.util.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileRecordReader;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;

public class BatchGenerationEngine extends Configured implements Tool {
	static Logger log = Logger.getLogger(BatchGenerationEngine.class);

	public static class BatchOutputFormat extends SequenceFileOutputFormat<Text,jBLASArrayWritable> {
		
	}
	
	public static class BatchInputFormat extends FileInputFormat<IntWritable,LabelImageWritable> {

		@Override
		public RecordReader<IntWritable, LabelImageWritable> createRecordReader(InputSplit is,
				TaskAttemptContext tac) throws IOException,
				InterruptedException {
			ImageReader ir = new ImageReader();
			ir.initialize(is,tac);
			return ir;
		}
		
	}
	public static class ImageReader extends RecordReader<IntWritable,LabelImageWritable> {
		private SequenceFileRecordReader<IntWritable, LabelImageWritable> reader;
		private IntWritable lineKey;
		private LabelImageWritable lineValue;		
		
		@Override
		public void close() throws IOException {
			reader.close();
		}

		@Override
		public IntWritable getCurrentKey() throws IOException, InterruptedException {
			return lineKey;
		}

		@Override
		public LabelImageWritable getCurrentValue() throws IOException, InterruptedException {
			return lineValue;
		}

		@Override
		public float getProgress() throws IOException, InterruptedException {
			return reader.getProgress();
		}

		@Override
		public void initialize(InputSplit is, TaskAttemptContext tac)
				throws IOException, InterruptedException {
			reader = new SequenceFileRecordReader<IntWritable, LabelImageWritable>();
			reader.initialize(is, tac);
		}

		@Override
		public boolean nextKeyValue() throws IOException, InterruptedException {
			if(!reader.nextKeyValue()) {
				return false;
			}
			lineKey = reader.getCurrentKey();
			lineValue = reader.getCurrentValue();
			return true;
		}
		
	}
	
	public static class Record {
		public int number;
		public int label;
		public byte[] data;
	}
	
	public static class ImageSplit extends Mapper<IntWritable,LabelImageWritable,IntWritable,jBLASArrayWritable> {
		private IntWritable sameKey = new IntWritable(0);
		private DoubleMatrix imageMat = null;
		private DoubleMatrix label = null;
		private jBLASArrayWritable output = null;
		
		@Override
		public void map(IntWritable key, LabelImageWritable value,
				ImageSplit.Context context) throws IOException, InterruptedException {
			byte[] image = value.getImage();
			int labelInt = value.getLabel();
			
			imageMat = new DoubleMatrix(1,image.length);
			label = new DoubleMatrix(1);
			
			label.put(0,labelInt);
			
			for(int i = 0; i < image.length; i++) {
				imageMat.put(0,i,image[i]);	
			}
			
			ArrayList<DoubleMatrix> list = new ArrayList<DoubleMatrix>();
			list.add(imageMat);
			list.add(label);
			output = new jBLASArrayWritable(list);
			context.write(sameKey, output);
			//log.info("Completed key "+key.toString());
		}
	}
	public static class Minibatcher extends Reducer<IntWritable,jBLASArrayWritable,Text,jBLASArrayWritable> {
		private Text batchID = new Text();
		private jBLASArrayWritable dataArray;
		
		private int layers = 1;
		private int visibleNodes = 28*28;
		private ArrayList<Integer> hiddenNodes = new ArrayList<Integer>();
		private int exampleCount = 60000;
		private int batchSize = 20;
		
		@Override
		public void setup(Minibatcher.Context context) {
			
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
						if(elName.compareToIgnoreCase("visible.nodes") == 0) {
							this.visibleNodes = Integer.parseInt(xmlGetSingleValue(property,"value"));
						} else if(elName.substring(0, 13).compareToIgnoreCase("hidden.nodes") == 0) {
							this.hiddenNodes.add(Integer.parseInt(xmlGetSingleValue(property,"value")));
						} else if(elName.compareToIgnoreCase("example.count") == 0) {
							this.exampleCount = Integer.parseInt(xmlGetSingleValue(property,"value"));
						} else if(elName.compareToIgnoreCase("batch.size") == 0) {
							this.batchSize = Integer.parseInt(xmlGetSingleValue(property,"value"));
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
		
		@Override
		public void reduce(IntWritable sameNum, Iterable<jBLASArrayWritable> data,
				Minibatcher.Context context) throws IOException, InterruptedException {
			
			int totalBatchCount = exampleCount/batchSize;
			Iterator<jBLASArrayWritable> dataIter = data.iterator();
			
			DoubleMatrix weights = DoubleMatrix.randn(hiddenNodes.get(0),visibleNodes);
			DoubleMatrix hbias = DoubleMatrix.zeros(1,hiddenNodes.get(0));
			DoubleMatrix vbias = DoubleMatrix.zeros(1,visibleNodes);
			DoubleMatrix label = DoubleMatrix.zeros(batchSize,1);
			DoubleMatrix hiddenChain = null;
			
			ArrayList<DoubleMatrix> outputmatricies = new ArrayList<DoubleMatrix>();
			outputmatricies.add(weights);
			outputmatricies.add(hbias);
			outputmatricies.add(hiddenChain);
			outputmatricies.add(vbias);
			outputmatricies.add(label);
			
			int j;
			for(int i = 1; i <= totalBatchCount; i++) {
				if(!dataIter.hasNext()) {
					break;
				}
				j = 0;
				DoubleMatrix vdata = DoubleMatrix.zeros(batchSize,visibleNodes);
				while(dataIter.hasNext() && j < batchSize ) {
					jBLASArrayWritable imageStore = dataIter.next();
					ArrayList<DoubleMatrix> list = imageStore.getData();
					vdata.putRow(j, list.get(0));
					label.put(j,list.get(1).get(0));
					j++;
				}
				outputmatricies.add(vdata);
				batchID.set(i+"");
				dataArray = new jBLASArrayWritable(cloneList(outputmatricies));
				context.write(batchID, dataArray);
			}
		}
		
		public static ArrayList<DoubleMatrix> cloneList(List<DoubleMatrix> list) {
			ArrayList<DoubleMatrix> clone = new ArrayList<DoubleMatrix>(list.size());
			for(DoubleMatrix item : list) {
				DoubleMatrix newItem = new DoubleMatrix();
				if (item != null) {
					clone.add(newItem.copy(item));
				}
				else {
					clone.add(null);
				}
					
			}
			return clone;
		}
		
	}
	
	@Override
	public int run(String[] args) throws Exception {
		Configuration conf = getConf();
		
		Job job = new Job(conf,"minibatch");
		
		job.setJarByClass(BatchGenerationEngine.class);
		FileInputFormat.setInputPaths(job, new Path(args[0]));
		FileOutputFormat.setOutputPath(job, new Path(args[1]));
		
		job.setMapperClass(ImageSplit.class);
		job.setReducerClass(Minibatcher.class);
		
		job.setNumReduceTasks(1);
		
		job.setInputFormatClass(BatchInputFormat.class);
				
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(jBLASArrayWritable.class);
		job.setMapOutputKeyClass(IntWritable.class);
		job.setMapOutputValueClass(jBLASArrayWritable.class);
		
		job.setOutputFormatClass(BatchOutputFormat.class);
		
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
		
		//distribute into different batch
				
		System.exit(result);
	}	
}
