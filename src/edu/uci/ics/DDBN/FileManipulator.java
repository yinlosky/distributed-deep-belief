
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.lang.String;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LabelImageWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.SequenceFile.Writer;

public class FileManipulator {

	/**
	 * @param args
	 * @throws IOException 
	 */
	public static void main(String[] args) throws IOException {
		String file_location = "/home/kraemahz/MNIST/train-images-idx3-ubyte";
		String label_location = "/home/kraemahz/MNIST/train-labels-idx1-ubyte";
		
		FileInputStream stream = new FileInputStream(file_location);
		FileInputStream labels = new FileInputStream(label_location);
						
		byte[] open_buffer = new byte[16];
		byte[] cruft_buffer = new byte[8];
		byte[] label_buffer = new byte[1];
		int magic;
		int count = 0;
		int rows = 0;
		int cols = 0;
		
		
		try {
			stream.read(open_buffer);
			labels.read(cruft_buffer);
			magic = byteArrayToInt(open_buffer,0);
			count = byteArrayToInt(open_buffer, 4);
			if(magic != 2051 || count != byteArrayToInt(cruft_buffer, 4)) {
				throw new IOException();
			}
			rows = byteArrayToInt(open_buffer,8);
			cols = byteArrayToInt(open_buffer, 12);		
			
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		System.out.println(count);
		System.out.println(rows);
		System.out.println(cols);
		
		byte[] image_buffer = new byte[rows*cols];
		
		Configuration conf = new Configuration();
	    conf.addResource(new Path("/home/hadoop/hadoop/conf/core-site.xml"));
	    conf.addResource(new Path("/home/hadoop/hadoop/conf/hdfs-site.xml"));
	    FileSystem fs = FileSystem.get(conf);
	    
		Path path = new Path("/user/hadoop/MNIST/training_images");
		try {
			Writer fileWriter = SequenceFile.createWriter(fs,conf,path,IntWritable.class,LabelImageWritable.class);
			for(int i = 0; i < count; i++) {
				stream.read(image_buffer);
				labels.read(label_buffer);
				fileWriter.append(new IntWritable(i),new LabelImageWritable((int)label_buffer[0],image_buffer,rows*cols));
			}
			fileWriter.close();			
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		
	}
	
	public static int byteArrayToInt(byte[] b, int offset) {
        int value = 0;
        for (int i = 0; i < 4; i++) {
            int shift = (4 - 1 - i) * 8;
            value += (b[i + offset] & 0x000000FF) << shift;
        }
        return value;
    }
}
