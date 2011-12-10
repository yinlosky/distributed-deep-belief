package edu.uci.ics.DDBN;

import java.awt.Graphics;
import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.io.IOException;
import java.util.ArrayList;

import javax.swing.JFrame;
import javax.swing.JPanel;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.jBLASArrayWritable;
import org.jblas.DoubleMatrix;

public class FileCheck {

	/**
	 * @param args
	 * @throws IOException 
	 */
	public static void main(String[] args) throws IOException {
		Configuration conf = new Configuration();
		FileSystem fs = FileSystem.getLocal(conf);
		Path check = new Path(args[0]);
		SequenceFile.Reader reader = 
			new SequenceFile.Reader(fs,check,conf);
		Text key = new Text();
		jBLASArrayWritable value = new jBLASArrayWritable();
		reader.next(key,value);
		ArrayList<DoubleMatrix> vals = value.getData();
		for(DoubleMatrix val : vals) {
			if(val != null) {
				System.out.println(val.rows + ", " + val.columns);
				if(val.length == 28*28) {
					val.reshape(28, 28);
					makeImage(val);
				} else if(val.length >= 500*28*28) {
					makeImage(val);
				}
			}
		}
	}

	public static void makeImage(DoubleMatrix mat) {
		JPanel draw = new WeightImage(mat);
		JFrame frame = new JFrame("Render " + mat.hashCode());
		frame.setSize(mat.columns + 50,mat.rows + 50);
		frame.getContentPane().add(draw);
		frame.setVisible(true);
	}
	
	public static class WeightImage extends JPanel {
		/**
		 * 
		 */
		private static final long serialVersionUID = 4154750361473682164L;
		private BufferedImage image;
		
		public WeightImage(DoubleMatrix mat) {
			this.image = new BufferedImage(mat.columns, mat.rows, BufferedImage.TYPE_BYTE_GRAY);
			WritableRaster r = image.getRaster();
			Double rescaleHigh = mat.max(), rescaleLow = mat.min();
			for(int i = 0; i < mat.columns; i++) {
				for(int j = 0; j < mat.rows; j++) {
					double[] pix = {255*(mat.get(j,i)-rescaleLow)/(rescaleHigh-rescaleLow)};
					r.setPixel(i, j, pix);
				}
			}
			image.setData(r);
		}
		
		@Override
		public void paintComponent(Graphics g) {
			int x = (this.getWidth()-image.getWidth())/2;
			int y = (this.getHeight()-image.getHeight())/2;
			g.drawImage(image, x, y, null);
		}
	}
	
}
