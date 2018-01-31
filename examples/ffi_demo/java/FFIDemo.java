// A Java example of calling ARL functions

import sdp.java.arljava;
import sdp.java.ARLConf;
import sdp.java.ARLVis;
import sdp.java.intArray;

public class FFIDemo {
    public static void main(String[] args) throws Exception {

	double cellsize = 0.0005;
	String config_name = "LOWBD2-CORE";	
	arljava.arl_initialize();


	ARLConf lowconfig=arljava.allocate_arlconf_default(config_name);;
	/*
	int nvis = lowconfig.getNbases()*lowconfig.getNfreqs()*lowconfig.getNtimes();

	ARLVis vt = arljava.allocate_vis_data(lowconfig.getNpol(), nvis);
	ARLVis vtmp = arljava.allocate_vis_data(lowconfig.getNpol(), nvis);

	intArray shape=new intArray(4);
	arljava.helper_get_image_shape(lowconfig.getFreqs(), cellsize,
				       shape.cast());
	*/
    }
}
