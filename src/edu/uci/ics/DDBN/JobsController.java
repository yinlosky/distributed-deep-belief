package edu.uci.ics.DDBN;

import java.io.IOException;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.StringUtils;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;


public class JobsController {
	public static Logger log = Logger.getLogger(BatchGenerationEngine.class);;
	public static JobConfig[] createJobsDic(Configuration conf, String input, String folder, int numJobs){
        JobConfig[] dictionary = new JobConfig[numJobs];
        //change tool here!
        dictionary[0]  = new JobConfig(conf,new WordCount(), input, folder + "/" + java.util.UUID.randomUUID().toString());
        for (int i = 1; i < numJobs; i++){
                dictionary[i] = new JobConfig(conf,new WordCount(), dictionary[i-1].args[1], folder + "/" + java.util.UUID.randomUUID().toString());
        }
        return dictionary;
}

	public static void RunJobs(JobConfig[] dictionary) throws Exception{
	        log.info("Start " + dictionary.length + " jobs!");
	        for (int i = 0; i < dictionary.length; i++){
	                int runResult = ToolRunner.run(dictionary[i].conf, dictionary[i].tool, dictionary[i].args);
	                if (runResult == 1){
	                        log.info("Job " + i + "-th Re-run once!");
	                        dictionary[i].args[1] = java.util.UUID.randomUUID().toString();
	                        runResult = ToolRunner.run(dictionary[i].conf, dictionary[i].tool, dictionary[i].args);
	                }
	                if (runResult == 1){
	                        log.info("Job " + i + "-th Re-run twice!");
	                        dictionary[i].args[1] = java.util.UUID.randomUUID().toString();
	                        runResult = ToolRunner.run(dictionary[i].conf, dictionary[i].tool, dictionary[i].args);
	                }
	                if (runResult == 1){
	                        log.info("Job " + i + "-th Failed!");
	                        break;
	                } else {
	                        // Update input of next job, since the current job failed!
	                        if (i + 1 < dictionary.length)
	                                dictionary[i + 1].args[0] = dictionary[i].args[1];
	                }
	        }
	}
	
	public static void distributeFiles(String input) throws IOException{
	        Configuration conf = new Configuration();
	        FileSystem fs = FileSystem.getLocal(conf);
	        Path path = new Path(input);
	        OutputSplitDir outmod = new OutputSplitDir(path,conf,fs);
	        outmod.execute();
	}
	
	public static int[] parseJobSetup(Path jobFile) {
	        int[] result = new int[2];
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
	                                if(elName == "example.count") {
	                                        result[0] = Integer.parseInt(xmlGetSingleValue(property,"value"));
	                                } else if(elName == "batch.size") {
	                                        result[1] = Integer.parseInt(xmlGetSingleValue(property,"value"));
	                                }
	                        }
	                }
	                
	        } catch (ParserConfigurationException pce) {
	                System.err.println("Caught exception while parsing the cached file '" + jobFile + "' : " + StringUtils.stringifyException(pce));
	                return null;
	        } catch(SAXException se) {
	                System.err.println("Caught exception while parsing the cached file '" + jobFile + "' : " + StringUtils.stringifyException(se));
	                return null;
	        }catch(IOException ioe) {
	                System.err.println("Caught exception while parsing the cached file '" + jobFile + "' : " + StringUtils.stringifyException(ioe));
	                return null;
	        }
	        return result;
	}
	
	private static String xmlGetSingleValue(Element el, String tag) {
	        return ((Element)el.getElementsByTagName(tag).item(0)).getFirstChild().getNodeValue();
	}
}
