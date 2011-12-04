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
import org.apache.hadoop.fs.permission.FsPermission;
import org.apache.hadoop.io.*;
import org.apache.hadoop.io.SequenceFile.*;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileRecordReader;
import org.jblas.DoubleMatrix;

public class OutputSplitDir {
	SequenceFile.Reader reader;
	SequenceFile.Writer writer;
	FileSystem fs;
	Configuration conf;
	Path dir;
	
	public static void main(String[] args) throws IOException {
		Configuration conf = new Configuration();
		FileSystem fs = FileSystem.getLocal(conf);
		Path path = new Path("/media/The_Universe/hadoop/tmp_hdfs/test");
		OutputSplitDir outmod = new OutputSplitDir(path,conf,fs);
		outmod.execute();
	}
	
	public OutputSplitDir(Path dir, Configuration conf, FileSystem fs) throws IOException{
		Path path = new Path(dir.toString()+"/part-r-00000");
		this.reader = new SequenceFile.Reader(fs,path,conf);
		this.writer = null;
		this.fs = fs;
		this.conf = conf;
		this.dir = dir;
	}
	
	public void execute() throws IOException{
		Text key = new Text();
		jBLASArrayWritable value = new jBLASArrayWritable(); 
		String prevBatch = "-1";
		while(reader.next(key,value)){
			String batchNum = key.toString().split(" ")[0];
			if(batchNum.compareToIgnoreCase(prevBatch)!=0){
				//make new subdirectory
				//set prevBatch = batchNum
				//create new file 
				prevBatch = batchNum;
				String dirName = dir.toString() + "/" + batchNum;
				Path newDir = new Path(dirName);
				fs.mkdirs(newDir);
				String fileName = dirName +"/" +batchNum;
				Path newFile = new Path(fileName);
				fs.createNewFile(newFile);
				if(writer != null) writer.close();
				this.writer = new SequenceFile.Writer(fs, conf, newFile, Text.class, jBLASArrayWritable.class);
			}
			//use sequenceFileWriter to write value into 
			writer.append(key, value);
			key = new Text();
			value = new jBLASArrayWritable(); 
		}
		reader.close();
		writer.close();
	}
}
