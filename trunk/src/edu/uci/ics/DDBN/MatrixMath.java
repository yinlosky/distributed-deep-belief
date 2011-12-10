package edu.uci.ics.DDBN;

import org.jblas.DoubleMatrix;

public class MatrixMath extends DoubleMatrix {
	
	/**
	 * Contains extra implementations of in-place element-wise matrix operations
	 * sigmoid - executes 1/(1-e^(-X))
	 * exp - executes e^(X)
	 * log - executes log(X)
	 * binom - returns 0.0/1.0 for each element taking its value to be p(Y=1)
	 * sum - sums the values into a row/column vector along the specified axis (0/1)
	 */
	private static final long serialVersionUID = 5088487362092770469L;

	public static DoubleMatrix sigmoid(DoubleMatrix exponent) {
		
		int rows = exponent.rows, cols = exponent.columns;
		for(int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				exponent.put(i,j, 1.0 / (1.0 + Math.exp(exponent.get(i, j)*-1.0)));
			}
		}			
		return exponent;
	}
	
	public static DoubleMatrix exp(DoubleMatrix exponent) {
		int rows = exponent.rows, cols = exponent.columns;
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < cols; j++) {
				exponent.put(i,j, Math.exp(exponent.get(i,j)));
			}
		}
		return exponent;
	}
	
	public static DoubleMatrix log(DoubleMatrix logponent) {
		int rows = logponent.rows, cols = logponent.columns;
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < cols; j++) {
				logponent.put(i,j, Math.log(logponent.get(i,j)));
			}
		}
		return logponent;
	}
	
	public static DoubleMatrix binom(DoubleMatrix p) {
		int rows = p.rows, cols = p.columns;
		for(int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				p.put(i,j, Math.random() < p.get(i,j) ? 1.0 : 0.0);
			}
		}	
		return p;
	}
	
	public static DoubleMatrix sum(DoubleMatrix summant, int axis ) {
		int rows = axis==0 ? summant.rows : 1, cols = axis==0 ? 1 : summant.columns;
		DoubleMatrix sum = new DoubleMatrix(rows,cols);
		if(axis == 0) {
			for(int i = 0; i < summant.columns; i++) {
				sum.put(1,i, summant.getColumn(i).sum());
			}
		} else {
			for(int i = 0; i < summant.rows; i++) {
				sum.put(i,1, summant.getRow(i).sum());
			}
		}
		return sum;
	}
}
