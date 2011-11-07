import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.URI;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.log4j.Logger;


public class ExtractImage {

	private static Logger logger = Logger.getLogger(ExtractImage.class);
	private static int imageWidth = 20;
	private static int imageHeight = 20;
	private static int overlap = 20; //which is 20%
	
	public static class ExtractImageMapper extends Mapper<Object, Text, Text, BytesWritable> {

		public void map(Object key, Text value, Context context) 
		throws IOException, InterruptedException {

			logger.info("map method called..");

			//the uri has this format:
			//imageFileURI LabelFileURI
			String[] uri = value.toString().split(" ");
			ImgObj[] listImgObj = getImage(testFileImage, testFileLabel);

			for(int i = 0; i< listImgObj.length; i++){
				context.write(listImgObj[i].id, new BytesWritable(listImgObj[i].toByteArray()));
			}

		}
		
		private ImgObj[] getImage(String imageFile, String labelFile) throws IOException {
			ImgObj[] testImg; 
			Configuration conf = new Configuration();
			FileSystem fs = FileSystem.get(URI.create(imageFile), conf);
			FSDataInputStream in = null;
			try {
				in = fs.open(new Path(imageFile));
				byte buffer[] = new byte[4];
				in.read(buffer, 4, 4);
				int numImg = byteArrayToInt(buffer, 0); 
				in.read(buffer, 8, 4);
				int rows = byteArrayToInt(buffer, 0); 
				in.read(buffer, 12, 4);
				int cols = byteArrayToInt(buffer, 0); 
				int startOff = 16;

				ImgObj[] testImg = getTestImagePixel(in, numImg, rows, cols, startOff);

			} finally {
				IOUtils.closeStream(in);
			}
		    

			Configuration conf1 = new Configuration();
			FileSystem fs1 = FileSystem.get(URI.create(labelFile), conf1);
			FSDataInputStream in1 = null;
		    
		    	try {
				in1 = fs.open(new Path(labelFile));
				byte buffer[] = new byte[4];
				in1.read(buffer, 4, 4);
				int numItems = byteArrayToInt(buffer, 0); 
				int startLabelOffset = 16;
				int[] Items = getLabel(in1,numItems, startLabelOffset);
				for(int i = 0; i< testImg.length; i++){
					testImg[i].label = Items[i];
				}
			} finally {
				IOUtils.closeStream(in1);
			}
		    
			return testImg;
		}
		
		  private ImgObj[] getTestImagePixel(FSDataInputStream in, int numImg, int row, int col, int startOff) throws IOException{
			  ImgObj[] testImgs = new ImgObj[numImg];
			  int size = row * col;
			  
			  int offset = startOff;
			  for(int i = 0; i < numImg; i++){
				  testImgs[i] = new ImgObj();
				  testImgs[i].id = i;
				  testImgs[i].pixels = new byte[size];
				  in.read(testImgs[i].pixels, startOff + i*size, size);
			  }
			  return testImgs;
		  }
		  
		  private int[] getLabel(FSDataInputStream in, int numItems, int startOff)throws IOException{
			  int[] items = new int[numItems];
			  int offset = startOff;
			  for(int i= 0;i<numItems;i++){
				byte buffer[] = new byte[1];
				in.read(buffer, ++offset, 1);
				byte buffer2[] = new byte[4];
				buffer2[3] = buffer[0];
				items[i] = byteArrayToInt(buffer2,0)
			  }
			  return items;
		  }
		  
		  public class ImgObj {
			public int id;
			public byte[] pixels;
			public int label;
		}
		
		private int byteArrayToInt(byte[] b, int offset) {
			int value = 0;
			for (int i = 0; i < 4; i++) {
			    int shift = (4 - 1 - i) * 8;
			    value += (b[i + offset] & 0x000000FF) << shift;
			}
			return value;
		}
		

	}


	public static void main(String[] args) throws Exception {
		Configuration conf = new Configuration();
		String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
		if (otherArgs.length != 2) {
			System.err.println("Usage: ExtractImage <in Path for url file> <out pat for sequence file>");
			System.exit(2);
		}
		
		if(otherArgs.length >= 5){
			imageWidth = Integer.parseInt( otherArgs[2] );;
			imageHeight = Integer.parseInt( otherArgs[3] );;
			overlap = Integer.parseInt( otherArgs[4] );; //which is 20%
		}
	
		Job job = new Job(conf, "ExtractImage");
		job.setJarByClass(ExtractImage.class);
		job.setMapperClass(ExtractImageMapper.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(BytesWritable.class);
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(SequenceFileOutputFormat.class);
		
		FileInputFormat.addInputPath(job, new Path(otherArgs[0]));
		FileOutputFormat.setOutputPath(job, new Path(otherArgs[1]));
		System.exit(job.waitForCompletion(true) ? 0 : 1);
	}


}