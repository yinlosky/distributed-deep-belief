package edu.uci.ics.DDBN;

import java.io.IOException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileRecordReader;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class OutputSplitDirMap extends Configured implements Tool {
		
	public static void main(String[] args) throws Exception {
		Configuration conf = new Configuration();
		ToolRunner.run(conf, new OutputSplitDirMap(), args);
	}

	@Override
	public int run(String[] args) throws Exception {
		
		Configuration conf = this.getConf();
		Job job = new Job(conf,"OutputSplit");
		
		job.setInputFormatClass(BatchFormat.class);
		
		job.setMapperClass(Mirror.class);
		
		job.setMapOutputKeyClass(Text.class);
		job.setMapOutputValueClass(jBLASArrayWritable.class);
		
		FileInputFormat.setInputPaths(job, new Path(args[0]));
		FileOutputFormat.setOutputPath(job, new Path(args[1]));
		
		job.setNumReduceTasks(0);
		
		return job.waitForCompletion(true) ? 0 : 1;
	}
	
	public static class BatchFormat extends FileInputFormat<Text,jBLASArrayWritable> {

		@Override
		public RecordReader<Text, jBLASArrayWritable> createRecordReader(
				InputSplit split, TaskAttemptContext context)
				throws IOException, InterruptedException {
			BatchReader br = new BatchReader();
			br.initialize(split,context);
			return br;
		}
		
	}
	
	public static class BatchReader extends RecordReader<Text,jBLASArrayWritable> {
		private SequenceFileRecordReader<Text, jBLASArrayWritable> reader;
		Text key;
		jBLASArrayWritable value;
		
		@Override
		public void close() throws IOException {
			reader.close();
		}

		@Override
		public Text getCurrentKey() throws IOException, InterruptedException {
			return key;
		}

		@Override
		public jBLASArrayWritable getCurrentValue() throws IOException,
				InterruptedException {
			return value;
		}

		@Override
		public float getProgress() throws IOException, InterruptedException {
			return reader.getProgress();
		}

		@Override
		public void initialize(InputSplit split, TaskAttemptContext context)
				throws IOException, InterruptedException {
			reader = new SequenceFileRecordReader<Text,jBLASArrayWritable>();
			key = new Text();
			value = new jBLASArrayWritable();
			reader.initialize(split, context);
		}

		@Override
		public boolean nextKeyValue() throws IOException, InterruptedException {
			if(!reader.nextKeyValue()) {
				return false;
			}
			key = new Text();
			value = new jBLASArrayWritable();			
			
			key = reader.getCurrentKey();
			value = reader.getCurrentValue();
			return true;
		}
		
	}
	
	public static class Mirror extends Mapper<Text,jBLASArrayWritable, Text,jBLASArrayWritable> {
		@Override
		public void map(Text key, jBLASArrayWritable value, Mirror.Context context) throws IOException, InterruptedException {
			context.write(key,value);
		}
	}
}
