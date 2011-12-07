package edu.uci.ics.DDBN;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.util.Tool;

public class JobConfig {
    Configuration conf;
    String[] args = null;
    Tool tool;
    public JobConfig(Configuration conf1,Tool tool1, String input1, String output1){
            conf = conf1;
            tool = tool1;
            args = new String[2];
            args[0] = input1;
            args[1] = output1;
    }
}
