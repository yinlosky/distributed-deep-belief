package org.apache.hadoop;


import main;

import java.io.*;
import java.util.*;
import java.lang.Math;

import org.apache.commons.math.linear.*;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapred.*;
import org.apache.hadoop.util.*;

public class BinarySigmoidRBM {
	
	@SuppressWarnings("deprecation")
	public static class MapClass extends MapReduceBase implements Mapper<LongWritable, MatrixArrayWritable, LongWritable, MatrixArrayWritable> {
		private RealMatrix W,b,c,v,h;
		
		public void map(LongWritable key, MatrixArrayWritable input, OutputCollector<LongWritable, MatrixArrayWritable> output, Reporter reporter) throws IOException {
			ArrayList<RealMatrix> data = input.getData();
			W = data.get(0);
			b = data.get(1);
			c = data.get(2);
			v = data.get(3);
			h = data.get(4);
			
			double v_by_b, h_by_c, W_by_vh, E, F;
			
			//calculate initial parameters
			v_by_b = v.preMultiply(b.transpose()).getEntry(0,0);
			h_by_c = h.preMultiply(c.transpose()).getEntry(0,0);
			W_by_vh = W.preMultiply(h.transpose()).multiply(v).getEntry(0,0);
			E = -v_by_b - h_by_c - W_by_vh;
			F = free_energy(v, b, c, W);
		}
		
		public static double free_energy(RealMatrix v, RealMatrix b, RealMatrix c, RealMatrix W) {
			double F = 0.0;
			double v_by_b = v.preMultiply(b.transpose()).getEntry(0,0);
			for(int i = 0; i < v.getColumnDimension(); i++)
				F -= Math.log(1. + Math.exp(c.getEntry(i, 0) + W.getColumnMatrix(i).multiply(v).getEntry(0,0)));
			F -= v_by_b;
			return F;
		}
		
		public static RealMatrix softmax(RealMatrix inputs, RealMatrix biases, RealMatrix weights) {
			double[] p_y_given_x = new double[inputs.getRowDimension()];
			double denominator = Math.exp(weights.multiply(inputs) + );
			
			for(int i = 0; i < inputs.getRowDimension(); i++) {
				p_y_given_x[i] = Math.exp(weights inputs[i] + biases[i]);
				denominator += p_y_given_x[i];
			}
			
			for(int i = 0; i < inputs.length; i++) {
				p_y_given_x[i] /= denominator;
			}
			
			return p_y_given_x;
		}
	}
}
