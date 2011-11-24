package edu.uci.ics.DDBN;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.util.ArrayList;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.Path;
import org.jblas.DoubleMatrix;

public class BatchWeightUpdate {
	private FileSystem fs;
	private Path updateDir;
	private Path updateFile;
	
	public BatchWeightUpdate(Path updateDir, Path updateFile) throws IOException {
		this(new Configuration(), updateDir, updateFile);
	}
	
	public BatchWeightUpdate(Configuration conf, Path updateDir, Path updateFile) throws IOException {
		this.setUpdateDir(updateDir);
		this.setUpdateFile(updateFile);
		this.setFs(FileSystem.get(conf));
	}

	public Configuration getConf() {
		return fs.getConf();
	}

	public FileSystem getFs() {
		return fs;
	}

	public void setFs(FileSystem fs) {
		this.fs = fs;
	}

	public Path getUpdateDir() {
		return updateDir;
	}

	public void setUpdateDir(Path path) {
		this.updateDir = path;
	}
	
	public Path getUpdateFile() {
		return updateFile;
	}

	public void setUpdateFile(Path path) {
		this.updateFile = path;
	}
	
	public boolean runUpdate() {
		try {
			DoubleMatrix[] updateMat = this.parseUpdateFile();
			return updateBatch(updateMat);
		} catch(IOException ioe) {
			System.err.println("Couldn't read update file " + ioe.toString());
			return false;
		}
	}
	
	private DoubleMatrix[] parseUpdateFile() throws IOException {
		FSDataInputStream in = fs.open(this.getUpdateFile());
		stripKey(in);
		DoubleMatrix[] ret = readArrayMatrix(in);
		in.close();
		return ret;
	}
	
	private boolean updateBatch(DoubleMatrix[] updateMat) {
		try {
			FileStatus[] listFiles = fs.listStatus(this.getUpdateDir());
			for(FileStatus status : listFiles) {
				Path fullName = status.getPath();
				FSDataInputStream in = fs.open(fullName);
				Path tempWriteFile = new Path(fullName.toString() +"-temp");
				FSDataOutputStream out = fs.create(tempWriteFile);
				try {
					while(true) {
						try{
							stripKey(in,out);
							DoubleMatrix[] matToUpdate = readArrayMatrix(in);
							matToUpdate[0] = updateMat[0]; //weights
							matToUpdate[1] = updateMat[1]; //hbias
							matToUpdate[2] = updateMat[2]; //vbias
							matToUpdate[5] = updateMat[3]; //hchain
							writeArrayMatrix(matToUpdate, out);
							out.writeChar('\n');
						} catch(EOFException eof) {
							break;
						}
					}
					fs.delete(fullName);
					fs.rename(tempWriteFile,fullName);
				} finally {
					in.close();
					out.close();
				}
			}
		} catch(IOException ioe) {
			System.err.println("Error in reading/writing from directory: " + ioe.toString());
			return false;
		}
		return true;
	}
	
	private void stripKey(FSDataInputStream in, FSDataOutputStream out) throws IOException, EOFException {
		String key = in.readUTF();
		out.writeUTF(key);
		char[] lastchar = new char[1];
		key.getChars(key.length()-1, key.length(), lastchar, 0);
		if(lastchar[0] != '\t') {
			//tab wasn't UTF, read a tab as ASCII
			char tab = in.readChar();
			out.writeChar(tab);
		}
	}
	
	private void stripKey(FSDataInputStream in) throws IOException, EOFException {
		String key = in.readUTF();
		char[] lastchar = new char[1];
		key.getChars(key.length()-1, key.length(), lastchar, 0);
		if(lastchar[0] != '\t') {
			//tab wasn't UTF, read a tab as ASCII
			char tab = in.readChar();
		}
	}
	
	private static DoubleMatrix[] readArrayMatrix(FSDataInputStream in) throws IOException {
		int count = in.readInt();
		System.out.println(count);
		DoubleMatrix[] ret = new DoubleMatrix[count];
		for(int i = 0; i < count; ++i) {
			ret[i] = readMatrix(in);
		}
		return ret;
	}
	
	private static DoubleMatrix readMatrix(FSDataInputStream in) throws IOException {
		int rows,cols;
		rows = in.readInt();
		cols = in.readInt();
		System.out.print(rows + " " + cols + "\n");
		double[][] temp = new double[rows][cols];
		for(int i = 0; i < rows; i++)
			for(int j = 0; j < cols; j++)
				temp[i][j] = in.readDouble();
		return new DoubleMatrix(temp);
	}
	
	private static void writeArrayMatrix(DoubleMatrix[] arrayMatrix, FSDataOutputStream out) throws IOException {
		out.writeInt(arrayMatrix.length);
		for(int i = 0; i < arrayMatrix.length; ++i) {
			writeMatrix(arrayMatrix[i], out);
		}
	}
	
	private static void writeMatrix(DoubleMatrix mat, FSDataOutputStream out) throws IOException {
		int rows = mat.getRows();
		int cols = mat.getColumns();
		out.writeInt(rows);
		out.writeInt(cols);
		for(int j = 0; j < rows; ++j) {
			for(int k = 0; k < cols; ++k) {
				out.writeDouble(mat.get(j,k));
			}
		}
	}
}
