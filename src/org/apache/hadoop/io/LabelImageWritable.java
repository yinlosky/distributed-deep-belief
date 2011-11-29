package org.apache.hadoop.io;
import org.apache.hadoop.io.Writable;
import java.io.*;

public class LabelImageWritable implements Writable {
	
	private int imageSize;
	private int label;
	private byte[] image;
	
	public LabelImageWritable(int label, byte[] image, int imageSize) {
		this.imageSize = imageSize;
		this.label = label;
		this.image = image;
	}
	
	public LabelImageWritable() {
		this.label = 0;
		this.imageSize = 28*28;
		this.image = new byte[imageSize];
	}
	
	@Override
	public void readFields(DataInput in) throws IOException {
		this.label = in.readInt();
		in.readChar(); //pull off separating tab
		for(int i = 0; i < imageSize; i++) {
			this.image[i] = in.readByte();
		}
	}
	
	@Override
	public void write(DataOutput out) throws IOException {
		out.writeInt(this.label);
		out.writeChar('\t');
		for(int i = 0; i < imageSize; i++) {
			out.writeByte(this.image[i]);
		}
	}
	
	public LabelImageWritable read(ObjectInputStream in) throws IOException, ClassNotFoundException {
		LabelImageWritable w = new LabelImageWritable();
		w.readFields(in);
		return w;
	}
	
	public byte[] getImage() {
		return this.image;	
	}
	
	public int getLabel() {
		return this.label;
	}
}
