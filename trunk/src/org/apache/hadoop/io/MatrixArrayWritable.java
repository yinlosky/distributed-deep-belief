package org.apache.hadoop.io;
import org.apache.hadoop.io.Writable;
import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;

import org.apache.commons.math.linear.*;

public class MatrixArrayWritable implements Writable {
	
	private ArrayList<RealMatrix> mlist;
	
	public MatrixArrayWritable(RealMatrix[] mlist) {
		this.mlist = new ArrayList<RealMatrix>(Arrays.asList(mlist));
	}
	
	public MatrixArrayWritable() {
		this.mlist = new ArrayList<RealMatrix>();
	}
	
	@Override
	public void readFields(DataInput in) throws IOException {
		int size = in.readInt();
		mlist.ensureCapacity(size);
		for(int i = 0; i < size; i++) {
			mlist.add(readMatrix(in));
		}
	}

	private RealMatrix readMatrix(DataInput in) throws IOException {
		int rows,cols;
		rows = in.readInt();
		cols = in.readInt();
		
		double[][] temp = new double[rows][cols];
		
		for(int i = 0; i < rows; i++)
			for(int j = 0; j < cols; j++)
				temp[i][j] = in.readDouble();
		return new Array2DRowRealMatrix(temp);
	}
	
	@Override
	public void write(DataOutput out) throws IOException {
		out.writeInt(mlist.size());
		for(RealMatrix M : mlist) {
			writeMatrix(out,M);
		}
	}
	
	private void writeMatrix(DataOutput out, RealMatrix M) throws IOException {
		int rows = M.getRowDimension(),cols = M.getColumnDimension();
		out.writeInt(rows);
		out.writeInt(cols);
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < cols; j++) {
				out.writeDouble(M.getEntry(rows, cols));
			}
		}
	}
	
	public MatrixArrayWritable read(ObjectInputStream in) throws IOException, ClassNotFoundException {
		MatrixArrayWritable w = new MatrixArrayWritable();
		w.readFields(in);
		return w;
	}
	
	public ArrayList<RealMatrix> getData() {
		return (ArrayList<RealMatrix>) mlist.clone();		
	}
}
