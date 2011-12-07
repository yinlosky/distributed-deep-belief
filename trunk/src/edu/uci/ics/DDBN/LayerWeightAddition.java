package edu.uci.ics.DDBN;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.jBLASArrayWritable;
import org.apache.hadoop.mapreduce.lib.input.InvalidInputException;
import org.apache.hadoop.util.StringUtils;
import org.jblas.DoubleMatrix;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

public class LayerWeightAddition extends BatchUpdater{
	
	public static void main(String[] args) throws IOException {
		Configuration conf = new Configuration();
		
		conf.setBoolean("minibatch.job.setup",true);
		DistributedCache.addCacheFile(new Path("/home/hadoop/batch-conf.xml").toUri(),conf);
		
		FileSystem fs = FileSystem.getLocal(conf);
		Path updateTo = new Path(args[0]);
		Path updateFrom = new Path(args[1]);
		LayerWeightAddition lwa = new LayerWeightAddition(fs,updateTo, updateFrom, 2);
		lwa.update();
	}

	private ArrayList<Integer> hiddenNodes;
	private int layer;
	private int layers;
	private int classCount;
	
	public LayerWeightAddition(Path updateTo, Path updateFrom, int layer)
		throws IOException {
		super(new Configuration(), updateTo, updateFrom);
		hiddenNodes = new ArrayList<Integer>();
		setLayer(layer);
		this.setup();
	}
	
	public LayerWeightAddition(Configuration conf, 
			Path updateTo, Path updateFrom, int layer) throws IOException {
		super(conf,updateTo,updateFrom);
		hiddenNodes = new ArrayList<Integer>();
		setLayer(layer);
		this.setup();
	}
	
	public LayerWeightAddition(FileSystem fs, Path updateTo, Path updateFrom, int layer) throws InvalidInputException {
		super(fs,updateTo,updateFrom);
		hiddenNodes = new ArrayList<Integer>();
		setLayer(layer);
		this.setup();
	}

	public void setLayer(int layer) throws InvalidInputException {
		if (layer < 2) {
			List<IOException> invalid = new ArrayList<IOException>();
			invalid.add(new IOException("Layer must be greater than 1"));
			throw new InvalidInputException(invalid);
		}
		this.layer = layer;
	}
	
	public int getLayer() {
		return this.layer;
	}
	
	@Override
	public void update() throws IOException {
		if (!fs.exists(updateTo) || !fs.exists(updateFrom)) {
			System.err.println(updateTo.getName() +
					" <- " + updateFrom.getName() +
					" : Not a valid update set");
			return;
		}
		
		SequenceFile.Reader reader_updateFrom = 
			new SequenceFile.Reader(fs,updateFrom,conf);
		SequenceFile.Reader reader_updateTo = 
			new SequenceFile.Reader(fs,updateTo,conf);
		Path writeLoc = new Path(updateTo.toString()+"-layer");
		fs.createNewFile(writeLoc);
		SequenceFile.Writer writer = 
			new SequenceFile.Writer(fs,conf,writeLoc,
					Text.class,jBLASArrayWritable.class);
		
		DoubleMatrix layerWeight, layerHbias, layerVbias,
					initWeight, initHbias, initVbias;
		Text key = new Text();
		jBLASArrayWritable value = new jBLASArrayWritable();
		
		reader_updateFrom.next(key, value);
		ArrayList<DoubleMatrix> updateData = value.getData();
		layerWeight = updateData.get(0);
		layerHbias = updateData.get(1);
		layerVbias = updateData.get(2);
		
		int visibleInput = layer == layers ? hiddenNodes.get(layer-1) + classCount 
										   : hiddenNodes.get(layer-1);
		
		initWeight = DoubleMatrix.randn(hiddenNodes.get(layer),visibleInput);
		initHbias = DoubleMatrix.zeros(1,hiddenNodes.get(layer));
		initVbias = DoubleMatrix.zeros(1,visibleInput);
		
		reader_updateFrom.close();
		
		while(reader_updateTo.next(key,value)) {
			updateData = value.getData();
			updateData.add(layerWeight);
			updateData.add(layerHbias);
			updateData.add(layerVbias);
			updateData.set(0,initWeight);
			updateData.set(1,initHbias);
			updateData.set(2,null);
			updateData.set(3,initVbias);			
			writer.append(key, new jBLASArrayWritable(updateData));
		}
		reader_updateTo.close();
		writer.close();
		
		fs.delete(updateTo,false);
		fs.rename(writeLoc, updateTo);
	}
	
	public void setup() {
		if (conf.getBoolean("minibatch.job.setup", false)) {
			Path[] jobSetupFiles = new Path[0];
			try {
				jobSetupFiles = DistributedCache.getLocalCacheFiles(conf);	
			} catch (IOException ioe) {
				System.err.println("Caught exception while getting cached files: " + StringUtils.stringifyException(ioe));
			}
			for (Path jobSetup : jobSetupFiles) {
				parseJobSetup(jobSetup);
			}
		}
	}

	private String xmlGetSingleValue(Element el, String tag) {
		return ((Element)el.getElementsByTagName(tag).item(0)).getFirstChild().getNodeValue();
	}
	
	private void parseJobSetup(Path jobFile) {
		DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
		try {
			DocumentBuilder db = dbf.newDocumentBuilder();
			Document doc = db.parse(jobFile.toString());
			Element configElement = doc.getDocumentElement();
			NodeList nodes = configElement.getElementsByTagName("property");
			if(nodes != null && nodes.getLength() > 0) {
				for(int i = 0; i < nodes.getLength(); i++) {
					Element property = (Element)nodes.item(i);
					String elName = xmlGetSingleValue(property,"name");
					if(elName.length() > 12 &&
							elName.substring(0, 12).compareToIgnoreCase("hidden.nodes") == 0) {
						this.hiddenNodes.add(Integer.parseInt(xmlGetSingleValue(property,"value")));
					} else if(elName.compareToIgnoreCase("layer.count") == 0) {
						this.layers = Integer.parseInt(xmlGetSingleValue(property,"value"));
					} else if(elName.compareToIgnoreCase("class.count") == 0) {
						this.classCount = Integer.parseInt(xmlGetSingleValue(property,"value"));
					}
				}
			}
			
		} catch (ParserConfigurationException pce) {
			System.err.println("Caught exception while parsing the cached file '" + jobFile + "' : " + StringUtils.stringifyException(pce));
		} catch(SAXException se) {
			System.err.println("Caught exception while parsing the cached file '" + jobFile + "' : " + StringUtils.stringifyException(se));
		}catch(IOException ioe) {
			System.err.println("Caught exception while parsing the cached file '" + jobFile + "' : " + StringUtils.stringifyException(ioe));
		}
	}
}
