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
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.InputFormat;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.OutputFormat;
import org.apache.hadoop.mapred.RecordReader;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.RunningJob;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.hadoop.mapred.SequenceFileRecordReader;
import org.apache.hadoop.mapred.TaskAttemptContext;



import org.apache.hadoop.util.*;



public class BatchGenerationEngine extends Configured implements Tool {
	static Logger log = Logger.getLogger(BatchGenerationEngine.class);

	@SuppressWarnings("deprecation")
	public static class BatchOutputFormat extends SequenceFileOutputFormat<Text,jBLASArrayWritable> {
		
	}
	
	@SuppressWarnings("deprecation")
	public static class BatchInputFormat extends FileInputFormat<IntWritable,LabelImageWritable> {

		public RecordReader<IntWritable, LabelImageWritable> createRecordReader(InputSplit is,
				TaskAttemptContext tac) throws IOException,
				InterruptedException {
			ImageReader ir = new ImageReader();
			ir.initialize(is,tac);
			return ir;
		}

		@Override
		public RecordReader<IntWritable, LabelImageWritable> getRecordReader(
				InputSplit arg0, JobConf arg1, Reporter arg2)
				throws IOException {
			// TODO Auto-generated method stub
			return null;
		}
		
	}
	public static class ImageReader implements RecordReader<IntWritable,LabelImageWritable> {
		private SequenceFileRecordReader<IntWritable, LabelImageWritable> reader;
		private IntWritable lineKey;
		private LabelImageWritable lineValue;		
		
		@Override
		public void close() throws IOException {
			reader.close();
		}

		public void initialize(InputSplit is, TaskAttemptContext tac) {
			// TODO Auto-generated method stub
			
		}

		public IntWritable getCurrentKey() throws IOException, InterruptedException {
			return lineKey;
		}

		public LabelImageWritable getCurrentValue() throws IOException, InterruptedException {
			return lineValue;
		}

		@Override
		public float getProgress() throws IOException {
			return reader.getProgress();
		}


		@Override
		public IntWritable createKey() {
			// TODO Auto-generated method stub
			return null;
		}

		@Override
		public LabelImageWritable createValue() {
			// TODO Auto-generated method stub
			return null;
		}

		@Override
		public long getPos() throws IOException {
			// TODO Auto-generated method stub
			return 0;
		}

		@Override
		public boolean next(IntWritable arg0, LabelImageWritable arg1)
				throws IOException {
			// TODO Auto-generated method stub
			return false;
		}
		
	}
	
	public static class Record {
		public int number;
		public int label;
		public byte[] data;
	}
	
	public static class ImageSplit extends MapReduceBase implements Mapper<IntWritable,LabelImageWritable,IntWritable,jBLASArrayWritable> {
		private IntWritable sameKey = new IntWritable(0);
		private DoubleMatrix imageMat = null;
		private DoubleMatrix label = null;
		private jBLASArrayWritable output = null;

		@Override
		public void map(IntWritable key, LabelImageWritable value,
				OutputCollector<IntWritable, jBLASArrayWritable> outputCollector,
				Reporter reporter) throws IOException {
			// TODO Auto-generated method stub
			log.info("Starting map");
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
			outputCollector.collect(sameKey, output);
			log.info("end mapper");
			
		}
	}
	public static class Minibatcher extends MapReduceBase implements Reducer<IntWritable,jBLASArrayWritable,Text,jBLASArrayWritable> {
		private Text batchID = new Text();
		private jBLASArrayWritable dataArray;
		private int visibleNodes = 28*28;
		private ArrayList<Integer> hiddenNodes = new ArrayList<Integer>();
		private int exampleCount = 60000;
		private int batchSize = 20;
		
		
		private String xmlGetSingleValue(Element el, String tag) {
			return ((Element)el.getElementsByTagName(tag).item(0)).getFirstChild().getNodeValue();
		}
		
		private void parseJobSetup(Path jobFile) {
			log.info("start parsing test.xml");
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
						} else if(elName.length() > 12 &&
								elName.substring(0, 12).compareToIgnoreCase("hidden.nodes") == 0) {
							this.hiddenNodes.add(Integer.parseInt(xmlGetSingleValue(property,"value")));
						} else if(elName.compareToIgnoreCase("example.count") == 0) {
							this.exampleCount = Integer.parseInt(xmlGetSingleValue(property,"value"));
						} else if(elName.compareToIgnoreCase("batch.size") == 0) {
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

		@Override
		public void reduce(IntWritable sameNum, Iterator<jBLASArrayWritable> data,
				OutputCollector<Text, jBLASArrayWritable> outputCollector, Reporter reporter)
				throws IOException {
			// TODO Auto-generated method stub
			
			log.info("reading test.xml");
			Path jobsetup = new Path("/user/kudo/test.xml");
			parseJobSetup(jobsetup);
			
			int totalBatchCount = exampleCount/batchSize;
			Iterator<jBLASArrayWritable> dataIter = data;
			
			DoubleMatrix weights = DoubleMatrix.randn(hiddenNodes.get(0),visibleNodes);
			DoubleMatrix hbias = DoubleMatrix.zeros(1,hiddenNodes.get(0));
			DoubleMatrix vbias = DoubleMatrix.zeros(1,visibleNodes);
			DoubleMatrix label = DoubleMatrix.zeros(batchSize,1);
			DoubleMatrix vdata = DoubleMatrix.zeros(batchSize,visibleNodes);
			DoubleMatrix hiddenChain = null;
			
			ArrayList<DoubleMatrix> outputmatricies = new ArrayList<DoubleMatrix>();
			outputmatricies.add(weights);
			outputmatricies.add(hbias);
			outputmatricies.add(hiddenChain);
			outputmatricies.add(vbias);
			outputmatricies.add(label);
			outputmatricies.add(vdata);
			
			int j;
			for(int i = 1; i <= totalBatchCount; i++) {
				if(!dataIter.hasNext()) {
					break;
				}
				j = 0;
				while(dataIter.hasNext() && j < batchSize ) {
					jBLASArrayWritable imageStore = dataIter.next();
					ArrayList<DoubleMatrix> list = imageStore.getData();
					vdata.putRow(j, list.get(0));
					label.put(j,list.get(1).get(0));
					j++;
				}
				batchID.set(i+"");
				dataArray = new jBLASArrayWritable(cloneList(outputmatricies));
				outputCollector.collect(batchID, dataArray);
			}
		}
		
	}
	
	@Override
	public int run(String[] args) throws Exception {

		JobConf conf = new JobConf(getConf(), BatchGenerationEngine.class); 
		conf.setJobName("minibatch");
		

		
		conf.setMapperClass(ImageSplit.class);
		conf.setReducerClass(Minibatcher.class);
		
		conf.setNumReduceTasks(1);
		conf.setInputFormat((Class<? extends InputFormat>) BatchInputFormat.class);
				
		conf.setOutputKeyClass(Text.class);
		conf.setOutputValueClass(jBLASArrayWritable.class);
		conf.setMapOutputKeyClass(IntWritable.class);
		conf.setMapOutputValueClass(jBLASArrayWritable.class);
		
		conf.setOutputFormat((Class<? extends OutputFormat>) BatchOutputFormat.class);
		
		FileInputFormat.setInputPaths(conf, new Path(args[0]));
		FileOutputFormat.setOutputPath(conf, new Path(args[1]));
		
		JobClient client = new JobClient(conf);
		RunningJob runningJob = client.submitJob(conf);
		log.info("Enter loop...");
//		while(!runningJob.isComplete()){
//			
//		};
		log.info("Exit loop...");
		
		return 0;
		//return job.waitForCompletion(true) ? 0 : 1;
	}
	
	public static void main(String[] args) throws Exception{
        System.out.println("Beginning job...");
        Configuration conf = new Configuration();
        String[] inputArgs = new GenericOptionsParser(conf,args).getRemainingArgs();
        
        Path xmlPath = null;
        List<String> other_args = new ArrayList<String>();
        for(int i = 0; i < args.length; ++i) {
                if("-setup".equals(inputArgs[i])) {
                        xmlPath = new Path(inputArgs[++i]);
                        //DistributedCache.addCacheFile(xmlPath.toUri(),conf);
                        conf.setBoolean("minibatch.job.setup",true);
                } else {
                        other_args.add(inputArgs[i]);
                }
        }
        
        String[] tool_args = other_args.toArray(new String[0]);
        int result = ToolRunner.run(conf, new BatchGenerationEngine(), tool_args);
        
//        log.info("distribute into different batch...");
        
//        JobsController.log = log;
//        JobsController.distributeFiles(tool_args[1]);
        
//        log.info("create Jobs...");
        // get example count and size from xml file
        // count = count_size[0];
        // size = count_size[1];
//        int[] count_size = JobsController.parseJobSetup(xmlPath);
//        int numJobs = count_size[1];
//        JobConfig[] dictionary = JobsController.createJobsDic(conf, tool_args[1]+"/1",tool_args[1], numJobs);
//        
//        log.info("Run Sequence Jobs...");
//        JobsController.RunJobs(dictionary);
        
        System.exit(result);
}



}
