package edu.uci.ics.DDBN;

import java.util.List;
import java.io.IOException;
import java.util.ArrayList;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.jBLASArrayWritable;
import org.jblas.DoubleMatrix;
import org.jblas.DoubleMatrix;

public class TestBin {

	/**
	 * @param args
	 * @throws IOException 
	 */
	public static void main(String[] args) throws IOException {
		ArrayList<Double> list = new ArrayList<Double>();
		list.add(10.0);
		list.add(20.0);
		list.add(30.0);
		List<Double> bit = list.subList(1, list.size());
		System.out.println(bit);
	}

}
