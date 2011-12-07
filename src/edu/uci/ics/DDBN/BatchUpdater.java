package edu.uci.ics.DDBN;

import java.io.IOException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

public abstract class BatchUpdater {
	protected FileSystem fs;
	protected Path updateTo;
	protected Path updateFrom;
	protected Configuration conf;
	
	public BatchUpdater(Path updateTo, Path updateFrom)
		throws IOException {
		this(new Configuration(), updateTo, updateFrom);
	}
	
	public BatchUpdater(Configuration conf, 
			Path updateTo, Path updateFrom) throws IOException {
		this.setUpdateFrom(updateFrom);
		this.setUpdateTo(updateTo);
		this.setConf(conf);
		this.setFs(FileSystem.get(conf));
	}
	
	public BatchUpdater(FileSystem fs, Path updateTo, Path updateFrom) {
		this.setUpdateFrom(updateFrom);
		this.setUpdateTo(updateTo);
		this.setConf(fs.getConf());
		this.setFs(fs);
	}
	
	public abstract void update() throws IOException;
	
	public void setConf(Configuration conf) {
		this.conf = conf;
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

	public Path getUpdateTo() {
		return updateTo;
	}

	public void setUpdateTo(Path path) {
		this.updateTo = path;
	}
	
	public Path getUpdateFrom() {
		return updateFrom;
	}

	public void setUpdateFrom(Path path) {
		this.updateFrom = path;
	}
}
