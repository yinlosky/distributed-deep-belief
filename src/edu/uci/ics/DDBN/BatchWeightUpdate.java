package edu.uci.ics.DDBN;

import java.io.IOException;
import java.util.ArrayList;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.Path;
import org.jblas.DoubleMatrix;

public class BatchWeightUpdate {
	
	void method() throws IOException {
		Configuration conf = new Configuration();
		FileSystem fs = FileSystem.get(conf);
		Path path = new Path("abc");
		
		path.getParent();
		
		fs.
	}
	
	
	
	public BatchWeightUpdate(Path updateDir, ArrayList<DoubleMatrix> updateMatricies) {
		
	}
	
}
