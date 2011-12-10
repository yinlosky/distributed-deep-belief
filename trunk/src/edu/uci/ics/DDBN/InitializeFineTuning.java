package edu.uci.ics.DDBN;

import java.io.IOException;
import java.util.ArrayList;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.jBLASArrayWritable;
import org.jblas.DoubleMatrix;

public class InitializeFineTuning extends BatchUpdater {

	private int epoch = 0;
	private int numLayers = 3;
	private int classCount = 10;
	public InitializeFineTuning(FileSystem fs, Path updateTo,
			Path updateFrom, int epoch, int numLayers, int classCount) throws IOException {
		super(fs,updateTo,updateFrom);
		setEpoch(epoch);
		setNumLayers(numLayers);
		setClassCount(classCount);
	}
	
	public InitializeFineTuning(FileSystem fs, Path updateTo,
			Path updateFrom, int epoch) throws IOException {
		super(fs,updateTo,updateFrom);
		setEpoch(epoch);
	}
	
	public InitializeFineTuning(Configuration conf, Path updateTo,
			Path updateFrom, int epoch, int numLayers, int classCount) throws IOException {
		super(conf, updateTo, updateFrom);
		setEpoch(epoch);
		setNumLayers(numLayers);
		setClassCount(classCount);
	}
	
	public InitializeFineTuning(Configuration conf, Path updateTo,
			Path updateFrom, int epoch) throws IOException {
		super(conf, updateTo, updateFrom);
		setEpoch(epoch);
	}
	
	public void setEpoch(int epoch) {
		this.epoch = epoch;
	}
	
	public int getEpoch() {
		return this.epoch;
	}
	
	public void setNumLayers(int numLayers) {
		this.numLayers = numLayers;
	}
	
	public int getNumLayers() {
		return this.numLayers;
	}
	
	public void setClassCount(int classCount) {
		this.classCount = classCount;
	}
	
	public int getClassCount() {
		return this.classCount;
	}

	@Override
	public void update() throws IOException {
		if (!fs.exists(updateFrom)) {
			System.err.println(updateFrom.getName() + " does not exist");
			return;
		}
		SequenceFile.Reader reader = new SequenceFile.Reader(fs, updateFrom, conf);
		
		try {
			fs.createNewFile(updateTo);
		} catch(IOException ioe) {
			System.err.println(updateTo.getName() + " already exists, overwriting");
			fs.delete(updateTo,false);
			fs.createNewFile(updateTo);
		}
		SequenceFile.Writer writer = new SequenceFile.Writer(fs,conf,updateTo,
				Text.class, jBLASArrayWritable.class);
		
		Text cached = new Text();
		jBLASArrayWritable result = new jBLASArrayWritable();
		
		//check if the records need to be split
		if(epoch == 1) {
			Path path = new Path(updateFrom.toString() + "-batches");
			fs.createNewFile(path);
			SequenceFile.Writer writerBatch = new SequenceFile.Writer(fs,conf,path,
					Text.class,jBLASArrayWritable.class);
			
			reader.next(cached,result);
			ArrayList<DoubleMatrix> data = result.getData();				
			DoubleMatrix[] weights = new DoubleMatrix[numLayers],
			hbias = new DoubleMatrix[numLayers+1];
			hbias[numLayers] = data.get(1);
			weights[numLayers-1] = data.get(0);
			int prelayer = (data.size()-6)/3;
			for(int i = 0; i < prelayer; i++) {
				weights[i] = data.get(6+i*3);
				if(i==0){
					hbias[1] = data.get(7);
					hbias[0] = data.get(8);
				} else {
					hbias[i+1] = data.get(7+i*3);
				}
			}
			DoubleMatrix classifierWeight = DoubleMatrix.randn(classCount,weights[numLayers-1].rows);
			DoubleMatrix classifierBias = DoubleMatrix.zeros(1,weights[numLayers-1].rows);
						
			ArrayList<DoubleMatrix> cacheList = new ArrayList<DoubleMatrix>();
			cacheList.add((new DoubleMatrix(1)).put(0, epoch));
			for(int i = 0; i < numLayers; i++) {
				cacheList.add(weights[i]);
			}
			for(int i = 0; i <= numLayers; i++) {
				cacheList.add(hbias[i]);
			}
			cacheList.add(classifierWeight);
			cacheList.add(classifierBias);
			
			jBLASArrayWritable cacheCollection = new jBLASArrayWritable(cacheList);
			writer.append(new Text("cached"), cacheCollection);
			DoubleMatrix v_data = data.get(5), label = data.get(4);
			ArrayList<DoubleMatrix> out= new ArrayList<DoubleMatrix>();
			out.add(label); out.add(v_data);
			writerBatch.append(new Text(""+(new Double(label.get(0))).intValue()),
					new jBLASArrayWritable(out));
			
			while(reader.next(cached,result)) {
				data = result.getData();
				v_data = data.get(5); label = data.get(4);
				out.set(0, label); out.set(1,v_data);
				writerBatch.append(new Text(""+(new Double(label.get(0))).intValue()),
						new jBLASArrayWritable(out));
			}
			
		} else {
			//else move the result into the cache
			while(reader.next(cached,result)) {
				if(cached.toString().compareToIgnoreCase("test") != 0) {
					ArrayList<DoubleMatrix> data = result.getData();
					data.get(0).put(0, epoch);
					writer.append(new Text("cached"), new jBLASArrayWritable(data));
					break;
				}
			}
		}
		reader.close();
		writer.close();
	}

	/**
	 * @param args
	 * @throws IOException 
	 */
	public static void main(String[] args) throws IOException {
		Configuration conf = new Configuration();
		FileSystem fs = FileSystem.getLocal(conf);
		Path updateTo = new Path(args[0]);
		Path updateFrom = new Path(args[1]);
		InitializeFineTuning ift = new InitializeFineTuning(fs,updateTo, updateFrom, 1,3,10);
		ift.update();
	}

}
