// A Java example of calling ARL functions

import sdp.java.arljava;
import sdp.java.ARLConf;
import sdp.java.ARLVis;
import sdp.java.Image;
import sdp.java.intArray;
import sdp.java.doubleArray;

public class FFIDemo {
    public static void main(String[] args) throws Exception {

	double cellsize = 0.0005;
	String config_name = "LOWBD2-CORE";	
	arljava.arl_initialize();


	ARLConf lowconfig=arljava.allocate_arlconf_default(config_name);;
	
	int nvis = lowconfig.getNbases()*lowconfig.getNfreqs()*lowconfig.getNtimes();

	System.out.println("nvis: "+ nvis );

	ARLVis vt = arljava.allocate_vis_data(lowconfig.getNpol(), nvis);

	
	ARLVis vtmp = arljava.allocate_vis_data(lowconfig.getNpol(), nvis);

	intArray shape=new intArray(4);
	arljava.helper_get_image_shape(lowconfig.getFreqs(), cellsize,
				       shape.cast());

	Image model = arljava.allocate_image(shape.cast());
	Image m31image = arljava.allocate_image(shape.cast());
	Image dirty = arljava.allocate_image(shape.cast());
	Image psf = arljava.allocate_image(shape.cast());
	Image comp = arljava.allocate_image(shape.cast());
	Image residual = arljava.allocate_image(shape.cast());
	Image restored = arljava.allocate_image(shape.cast());

	arljava.arl_create_visibility(lowconfig, vt);

	arljava.arl_create_test_image(lowconfig.getFreqs(),
				      cellsize,
				      vt.getPhasecentre(),
				      m31image);


	arljava.arl_predict_2d(vt, m31image, vtmp);
	vt = arljava.destroy_vis(vt);
	vt = vtmp;

	// BN: Why??
	// vtmp = NULL;

	arljava.arl_create_image_from_visibility(vt, model);

	doubleArray sumwt=new doubleArray(1);
	arljava.arl_invert_2d(vt, model, false, dirty, sumwt.cast());
	arljava.arl_invert_2d(vt, model, true, psf, sumwt.cast());

	arljava.arl_deconvolve_cube(dirty, psf, comp, residual);
	arljava.arl_restore_cube(comp, psf, residual, restored);

	// FITS files output

	// BN : to be done in JAVA:
	// status = mkdir("results", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	
	int status = arljava.export_image_to_fits_c(m31image, "results/m31image.fits");
	status = arljava.export_image_to_fits_c(dirty, "results/dirty.fits");
	status = arljava.export_image_to_fits_c(psf, "results/psf.fits");
	status = arljava.export_image_to_fits_c(residual, "results/residual.fits");
	status = arljava.export_image_to_fits_c(restored, "results/restored.fits");
	status = arljava.export_image_to_fits_c(comp, "results/solution.fits");

	model = arljava.destroy_image(model);
	m31image = arljava.destroy_image(m31image);
	dirty = arljava.destroy_image(dirty);
	psf = arljava.destroy_image(psf);
	residual = arljava.destroy_image(residual);
	restored = arljava.destroy_image(restored);


	ARLVis vtmodel = arljava.allocate_vis_data(lowconfig.getNpol(), nvis);
	vtmp = arljava.allocate_vis_data(lowconfig.getNpol(), nvis);

	arljava.arl_create_visibility(lowconfig, vtmodel);

	arljava.arl_predict_2d(vtmodel, comp, vtmp);

	vtmodel = arljava.destroy_vis(vtmodel);
	vtmp = arljava.destroy_vis(vtmp);

	comp = arljava.destroy_image(comp);

	arljava.arl_finalize();


	System.out.println("Configuration: " + config_name);
    }
}
