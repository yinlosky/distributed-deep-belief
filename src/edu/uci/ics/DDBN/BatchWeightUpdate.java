package edu.uci.ics.DDBN;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.jBLASArrayWritable;
import org.jblas.DoubleMatrix;

public class BatchWeightUpdate extends BatchUpdater {
	
	public static void main(String[] args) throws IOException {
		Configuration conf = new Configuration();
		FileSystem fs = FileSystem.getLocal(conf);
		Path updateTo = new Path(args[0]);
		Path updateFrom = new Path(args[1]);
		BatchWeightUpdate bwu = new BatchWeightUpdate(fs,updateTo, updateFrom);
		bwu.update();
	}
	
	public BatchWeightUpdate(Path updateTo, Path updateFrom)
		throws IOException {
		super(new Configuration(), updateTo, updateFrom);
	}
	
	public BatchWeightUpdate(Configuration conf, 
			Path updateTo, Path updateFrom) throws IOException {
		super(conf,updateTo,updateFrom);
	}
	
	public BatchWeightUpdate(FileSystem fs, Path updateTo, Path updateFrom) {
		super(fs,updateTo,updateFrom);
	}
	
	@Override
	public void update() throws IOException {
		if (!fs.exists(updateTo) || !fs.exists(updateFrom)) {
			System.err.println(updateTo.getName() +
					" <- " + updateFrom.getName() +
					" : File does not exist");
			return;
		}
		
		SequenceFile.Reader reader_updateFrom = 
			new SequenceFile.Reader(fs,updateFrom,conf);
		SequenceFile.Reader reader_updateTo = 
			new SequenceFile.Reader(fs,updateTo,conf);
		Path writeLoc = new Path(updateTo.toString()+"-up");
		fs.createNewFile(writeLoc);
		SequenceFile.Writer writer = 
			new SequenceFile.Writer(fs,conf,writeLoc,
					Text.class,jBLASArrayWritable.class);
		
		DoubleMatrix newWeight, newHbias, newVbias;		
		Text key = new Text();
		jBLASArrayWritable value = new jBLASArrayWritable();
		
		reader_updateFrom.next(key, value);
		ArrayList<DoubleMatrix> updateData = value.getData();
		newWeight = updateData.get(0);
		newHbias = updateData.get(1);
		newVbias = updateData.get(2);
		List<DoubleMatrix> hiddenChains = updateData.subList(3, updateData.size());
		
		reader_updateFrom.close();
		int i = 0;
		while(reader_updateTo.next(key,value)) {
			updateData = value.getData();
			updateData.get(0).copy(newWeight);
			updateData.get(1).copy(newHbias);
			if(updateData.get(2)==null) updateData.set(2,hiddenChains.get(i));
			else updateData.get(2).copy(hiddenChains.get(i));
			updateData.get(3).copy(newVbias);
			writer.append(key, new jBLASArrayWritable(updateData));
			i++;
		}
		reader_updateTo.close();
		writer.close();
		
		fs.delete(updateTo,false);
		fs.rename(writeLoc, updateTo);
	}	
}
