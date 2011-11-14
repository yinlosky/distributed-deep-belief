package org.apache.hadoop;


import java.io.*;
import java.lang.*;
import java.util.*;
import java.lang.Math;

import org.jblas.*;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapred.*;
import org.apache.hadoop.util.*;

public class BinarySigmoidRBM {
	
	public static final int PCD_STEPS = 1;
	public static final double LEARNING_RATE = 0.1;
	
	@SuppressWarnings("deprecation")
	public static class MapClass extends MapReduceBase implements Mapper<LongWritable, jBLASArrayWritable, LongWritable, jBLASArrayWritable> {
		private DoubleMatrix weights,hbias,vbias;
		private List<DoubleMatrix> hidden_state,v_data;
		private DoubleMatrix v1_data,h1_data;
		private DoubleMatrix w1,hb1,vb1;
		
		@Override
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
				for(int j = 0; j < PCD_STEPS; j++) {
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
			this.w1.add(v0.transpose().mul(h0).sub(v1.transpose().mul(h1)).mul(LEARNING_RATE));
			this.vb1.add(v0.sub(v1).mul(LEARNING_RATE));
			this.hb1.add(h0.sub(h1).mul(LEARNING_RATE));
		}
	}
	
	public static class ReduceClass extends MapReduceBase implements Reducer<LongWritable, jBLASArrayWritable, LongWritable, jBLASArrayWritable> {

		@Override
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
}
