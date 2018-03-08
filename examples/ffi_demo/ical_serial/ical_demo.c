#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <math.h>

#include "../include/arlwrap.h"
#include "../include/wrap_support.h"

/*
 * Verifies that:
 * - vt and vtmp are unique in memory
 * - vt and vtmp have equivalent values
 */
int verify_arl_copy(ARLVis *vt, ARLVis *vtmp)
{
	char *vtdata_bytes, *vtmpdata_bytes;
	int ARLVisDataSize;
	int i;

	if (vt == vtmp) {
		fprintf(stderr, "vt == vtmp\n");
		return 1;
	}

	if (!((vt->nvis == vtmp->nvis) && (vt->npol == vtmp->npol))) {
		return 2;
	}

	if (vt->data == vtmp->data) {
		return 3;
	}

	ARLVisDataSize = 80 + (32 * vt->npol * vt->nvis);
	vtdata_bytes = (char*) vt->data;
	vtmpdata_bytes = (char*) vtmp->data;

	for (i=0; i<ARLVisDataSize; i++) {
		if (vtdata_bytes[i] != vtmpdata_bytes[i]) {
			return 4;
		}
	}

	return 0;
}

int main(int argc, char **argv)
{
	int *shape  = malloc(4*sizeof(int));
	int *shape1 = malloc(4*sizeof(int));
	int status;
	int nvis;

	// ICAL section
	int wprojection_planes, i, nmajor, first_selfcal;
	double fstart, fend, fdelta, tstart, tend, tdelta, rmax, thresh;
	ARLadvice adv;
	ant_t nb;			//nant and nbases
	long long int *cindex_predict, *cindex_ical, *cindex2_ical;
	int cindex_nbytes;
	ARLGt *gt;			//GainTable
	ARLGt *gt_ical;			//GainTable (ICAL unrolled)
	bool unrolled = true; // true - unrolled, false - arl.functions::ical()
	// end ICAL section

	double cellsize = 0.0005;
	char config_name[] = "LOWBD2-CORE";
	char pol_frame [] = "stokesI";

	ARLVis *vt;			//Blockvisibility
	ARLVis *vtmodel;
	ARLVis *vtmp;
	ARLVis *vtpredicted;		//Visibility
	ARLVis *vt_predictfunction;	//Blockvisibility
	ARLVis *vt_gt;			//Blockvisibility
	ARLVis *vis_ical, *vpred_ical;	//Visibility ICAL temp
	ARLVis *vres_ical;		//Visibility ICAL temp
	ARLVis *bvtmp_ical, *bvtmp2_ical, *bvpred_ical;//Blockvisibility ICAL temp

	ARLConf *lowconfig;

	Image *gleam_model;		//Image (GLEAM model)
	Image *model;			//Image (a model for CLEAN)
	Image *dirty;			//Image (dirty image by invert_function)
	Image *deconvolved;		//Image (ICAL result)
	Image *residual;		//Image (ICAL result)
	Image *restored;		//Image (ICAL result)
	Image *psf_ical;		//Image (ICAL unrolled psf)
	Image *dirty_ical;		//Image (ICAL unrolled dirty)
	Image *cc_ical;			//Image (ICAL unrolled temp restored)
	Image *res_ical;		//Image (ICAL unrolled temp residuals)


	arl_initialize();

	lowconfig = allocate_arlconf_default(config_name);

	// ICAL section
	lowconfig->polframe = pol_frame;
	lowconfig->rmax = 300.0;
	// Get new nant and nbases w.r.t. to a maximum radius rmax
	helper_get_nbases_rmax(config_name, lowconfig->rmax, &nb);
	lowconfig->nant = nb.nant;
	lowconfig->nbases = nb.nbases;
	// Overwriting default values for a phasecentre
	lowconfig->pc_ra = 30.0;					// Phasecentre RA
	lowconfig->pc_dec = -60.0;					// Phasecentre Dec
	// Setting values for the frequencies and times
	lowconfig->nfreqs = 5; 						// Number of frequencies
	lowconfig->nchanwidth = lowconfig->nfreqs;			// Number of channel bandwidths
	fstart = 0.8e8;							// Starting frequency
	fend = 1.2e8;							// Ending frequency
	fdelta = (fend - fstart)/ (double)(lowconfig->nfreqs - 1);	// Frequency step
	lowconfig->ntimes = 11;						// Number of the times
	tstart = -M_PI/3.0;						// Starting time (in radians)
	tend = M_PI/3.0;						// Ending time (in radians)
	tdelta = (tend - tstart)/(double)(lowconfig->ntimes - 1);	// Time between the snapshots
	// Overwrining defalut frequency list
	free(lowconfig->freqs);
	free(lowconfig->channel_bandwidth);
	lowconfig->freqs = malloc(lowconfig->nfreqs * sizeof(double));
	lowconfig->channel_bandwidth = malloc(lowconfig->nfreqs * sizeof(double));
	printf("Frequency and bandwidth list\n");
	for(i = 0; i < lowconfig->nfreqs; i++) {
		lowconfig->freqs[i] = fstart + (double)i*fdelta;		
		lowconfig->channel_bandwidth[i] = fdelta;
		printf("%d %e %e\n", i,lowconfig->freqs[i], lowconfig->channel_bandwidth[i] );
		}
	// Overwriting default time list
	free(lowconfig->times);
	lowconfig->times = calloc(lowconfig->ntimes, sizeof(double));	
	printf("\nA list of the times (in rad)\n");
	for(i = 0; i < lowconfig->ntimes; i++) {
		lowconfig->times[i] = tstart + (double)i*tdelta;		
		printf("%d %e\n", i,lowconfig->times[i]);
		}
	// end ICAL section

	nvis = (lowconfig->nbases)*(lowconfig->nfreqs)*(lowconfig->ntimes);
	printf("Nvis = %d\n", nvis);
	
//	vt = allocate_vis_data(lowconfig->npol, nvis);
//	vtmp = allocate_vis_data(lowconfig->npol, nvis);
	vt 		   = allocate_blockvis_data(lowconfig->nant, lowconfig->nfreqs, lowconfig->npol, lowconfig->ntimes); //Blockvisibility
	vt_predictfunction = allocate_blockvis_data(lowconfig->nant, lowconfig->nfreqs, lowconfig->npol, lowconfig->ntimes); //Blockvisibility
	vt_gt              = allocate_blockvis_data(lowconfig->nant, lowconfig->nfreqs, lowconfig->npol, lowconfig->ntimes); //Blockvisibility
	vtpredicted        = allocate_vis_data(lowconfig->npol, nvis);							     //Visibility

	// Allocating cindex array where 8*sizeof(char) is sizeof(python int)
	cindex_nbytes = lowconfig->ntimes * lowconfig->nant * lowconfig->nant * lowconfig->nfreqs * sizeof(long long int);
	
	if (!(cindex_predict = malloc(cindex_nbytes))) {
		free(cindex_predict);
		return 1;
	}
	
	// ICAL section	
	// create_blockvisibility()
	printf("Create blockvisibility... ");
	arl_create_blockvisibility(lowconfig, vt);
	printf("Nrec = %d\n", lowconfig->nrec);
	// Allocating gaintable data
	gt = allocate_gt_data(lowconfig->nant, lowconfig->nfreqs, lowconfig->nrec, lowconfig->ntimes);

	// adwise_wide_field()
	adv.guard_band_image = 4.0;
	adv.delA=0.02;
	adv.wprojection_planes = 1;
	printf("Calculating wide field parameters... ");
	arl_advise_wide_field(lowconfig, vt, &adv);
	printf("Done.\n");
	printf("Vis_slices = %d,  npixel = %d, cellsize = %e\n", adv.vis_slices, adv.npixel, adv.cellsize);
	cellsize = adv.cellsize;

	// create_low_test_image_from_gleam
	helper_get_image_shape_multifreq(lowconfig, adv.cellsize, adv.npixel, shape);
	printf("A shape of the modeled GLEAM image: [ %d, %d, %d, %d]\n", shape[0], shape[1], shape[2], shape[3]);
	gleam_model = allocate_image(shape);
	arl_create_low_test_image_from_gleam(lowconfig, adv.cellsize, adv.npixel, vt->phasecentre, gleam_model);

	// FITS file output
	status = mkdir("results", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	status = export_image_to_fits_c(gleam_model, "!results/gleam_model.fits");

	// predict_function()
	arl_predict_function(lowconfig, vt, gleam_model, vtpredicted, vt_predictfunction, cindex_predict);

	// convert_visibility_to_blockvisibility()
	arl_convert_visibility_to_blockvisibility(lowconfig, vtpredicted, vt_predictfunction, cindex_predict, vt);

	// create_gaintable_from_blockvisibility()
	arl_create_gaintable_from_blockvisibility(lowconfig, vt, gt);

	// simulate_gaintable()
	arl_simulate_gaintable(lowconfig, gt);

	// apply_gaintable()
	arl_apply_gaintable(lowconfig, vt, gt, vt_gt, 1);

	// create_image_from_blockvisibility()
	// Create an image with nchan = 1
	for(i = 0; i< 4; i++) {
		shape1[i] = shape[i];
		}
	shape1[0] = 1;
	model = allocate_image(shape1);
	arl_create_image_from_blockvisibility(lowconfig, vt, adv.cellsize, adv.npixel, vt->phasecentre, model);

	// invert_function()
	dirty = allocate_image(shape1);
	arl_invert_function(lowconfig, vtpredicted, model, adv.vis_slices, dirty);

	// FITS file output
	status = export_image_to_fits_c(dirty, "!results/dirty.fits");

	// ical() - serial version
	deconvolved = allocate_image(shape1);
	residual    = allocate_image(shape1);
	restored    = allocate_image(shape1);

	if(unrolled){
	// The same values as hard-coded in arlwrap.py calls
		nmajor = 5; 
		thresh = 0.1;
		first_selfcal = 1;
		printf("ical: Performing %d major cycles\n", nmajor);
	// Allocate temp objects
		vis_ical  = allocate_vis_data(lowconfig->npol, nvis);							     //ICAL Visibility object (ical.vis)
		vpred_ical  = allocate_vis_data(lowconfig->npol, nvis);							     //ICAL Visibility object (ical.vispred)
		vres_ical  = allocate_vis_data(lowconfig->npol, nvis);							     //ICAL Visibility object (ical.visres)
		bvtmp_ical = allocate_blockvis_data(lowconfig->nant, lowconfig->nfreqs, lowconfig->npol, lowconfig->ntimes); //Blockvisibility vis_ical.blockvis (ical.vis.blockvis)
		bvtmp2_ical = allocate_blockvis_data(lowconfig->nant, lowconfig->nfreqs, lowconfig->npol, lowconfig->ntimes); //Blockvisibility vpred_ical.blockvis (ical.vispred.blockvis)
		bvpred_ical = allocate_blockvis_data(lowconfig->nant, lowconfig->nfreqs, lowconfig->npol, lowconfig->ntimes); //Blockvisibility (ical.block_vispred)
		psf_ical    = allocate_image(shape1);									// Image PSF
		dirty_ical    = allocate_image(shape1);									// Image dirty (CLEAN loop)
		cc_ical    = allocate_image(shape1);									// Image restored tmp (CLEAN loop)
		res_ical    = allocate_image(shape1);									// Image residual tmp (CLEAN loop)
		if (!(cindex_ical = malloc(cindex_nbytes))) {								     // Cindex vtmp_ical.cindex (ical.vis.cindex)
			free(cindex_ical);
			return 1;
		}
		if (!(cindex2_ical = malloc(cindex_nbytes))) {								     // Cindex vtmp_ical.cindex (ical.vis.cindex)
			free(cindex2_ical);
			return 1;
		}

		gt_ical = allocate_gt_data(lowconfig->nant, lowconfig->nfreqs, lowconfig->nrec, lowconfig->ntimes);	// GainTable (CLEAN loop)

	// convert_blockvisibility_to_visibility()
		arl_convert_blockvisibility_to_visibility(lowconfig, vt_gt, vis_ical, cindex_ical, bvtmp_ical);

	// copy_visibility (blockvis)
		arl_copy_blockvisibility(lowconfig, vt_gt, bvpred_ical, 0);

	// convert_blockvisibility_to_visibility()
	// Re-using cindex_ical, bvtmp_ical
		arl_convert_blockvisibility_to_visibility(lowconfig, bvpred_ical, vpred_ical, cindex2_ical, bvtmp2_ical);

	// Set vpred_ical.data to zero
		arl_set_visibility_data_to_zero(lowconfig, vpred_ical);

	// copy_visibility (vis)
		arl_copy_visibility(lowconfig, vpred_ical, vres_ical, 1);

	// predict_function()
		arl_predict_function_ical(lowconfig, vpred_ical, model, bvtmp2_ical, cindex2_ical, adv.vis_slices);

	// convert_visibility_to_blockvisibility()
		arl_convert_visibility_to_blockvisibility(lowconfig, vpred_ical, bvtmp2_ical, cindex2_ical, bvpred_ical);

	// Subtract visibility data to find residuals
        // vres = vtmp - vpred : 0 = add, 1 = subtract, 2 = mult, 3 = divide, else sets to zero
		arl_manipulate_visibility_data(lowconfig, vis_ical, vpred_ical, vres_ical, 1); 
	// arl_invert_function_ical() (extra parameters in **kwargs -TBS later)
		arl_invert_function_ical(lowconfig, vres_ical, model, adv.vis_slices, dirty_ical);
	// arl_invert_function_psf() (extra parameters in **kwargs -TBS later)
		arl_invert_function_psf(lowconfig, vres_ical, model, adv.vis_slices, psf_ical);
	// CLEAN major cycles		
		for(i = 0; i< nmajor; i++){
			printf("ical: Start of major cycle %d of %d\n", i, nmajor);
		// arl_deconvolve_cube_ical() with hard-coded **kwargs	
			arl_deconvolve_cube_ical(dirty_ical, psf_ical, cc_ical, res_ical);
		// add cc_ical into the model
			arl_add_to_model(model, cc_ical);
		// Set vpred_ical.data to zero
			arl_set_visibility_data_to_zero(lowconfig, vpred_ical);
		// predict_function()
			arl_predict_function_ical(lowconfig, vpred_ical, model, bvtmp2_ical, cindex2_ical, adv.vis_slices);
		// if doselfcal
			if(i >= first_selfcal) {
				printf("ical: Performing selfcalibration\n");
				// convert_visibility_to_blockvisibility()
				arl_convert_visibility_to_blockvisibility(lowconfig, vpred_ical, bvtmp2_ical, cindex2_ical, bvpred_ical);
				// arl_solve_gaintable()
				arl_solve_gaintable_ical(lowconfig, vt_gt, bvpred_ical, gt_ical, adv.vis_slices);
				// arl_apply_gaintable_ical() and re-write vt_gt (ical::block_vis)
				arl_apply_gaintable_ical(lowconfig, vt_gt, gt_ical, 0);
				// convert_blockvisibility_to_visibility()
				arl_convert_blockvisibility_to_visibility(lowconfig, vt_gt, vis_ical, cindex_ical, bvtmp_ical);
			}
			
        	// vres = vtmp - vpred : 0 = add, 1 = subtract, 2 = mult, 3 = divide, else sets to zero
			arl_manipulate_visibility_data(lowconfig, vis_ical, vpred_ical, vres_ical, 1); 
		// arl_invert_function_ical() (extra parameters in **kwargs -TBS later)
			arl_invert_function_ical(lowconfig, vres_ical, model, adv.vis_slices, dirty_ical);

		// ToDo - loop break on threshold

			printf("ical: End of major cycle %d\n", i);

		}
		printf("ical: End of major cycles\n");
		arl_restore_cube_ical(model, psf_ical, dirty_ical, restored);

	// FITS file output
		status = export_image_to_fits_c(model, 		"!results/deconvolved.fits");
		status = export_image_to_fits_c(dirty_ical,	"!results/residual.fits");
		status = export_image_to_fits_c(restored, 	"!results/restored.fits");
	// Cleaning up temp objects	
		bvtmp_ical 		= destroy_vis(bvtmp_ical);
		bvtmp2_ical 		= destroy_vis(bvtmp2_ical);
		bvpred_ical		= destroy_vis(bvpred_ical);
		vis_ical 		= destroy_vis(vis_ical);
		vpred_ical 		= destroy_vis(vpred_ical);
		vres_ical 		= destroy_vis(vres_ical);
		psf_ical		= destroy_image(psf_ical);
		dirty_ical		= destroy_image(dirty_ical);
		cc_ical			= destroy_image(cc_ical);
		res_ical		= destroy_image(res_ical);
		gt_ical 		= destroy_gt(gt_ical);

		free(cindex_ical);
		free(cindex2_ical);

	} else {	
		arl_ical(lowconfig, vt_gt, model, adv.vis_slices, deconvolved, residual, restored);
	// FITS file output
		status = export_image_to_fits_c(deconvolved, 	"!results/deconvolved.fits");
		status = export_image_to_fits_c(residual, 	"!results/residual.fits");
		status = export_image_to_fits_c(restored, 	"!results/restored.fits");
	}



	// Cleaning up
	gleam_model 	= destroy_image(gleam_model);
	model		= destroy_image(model);
	dirty		= destroy_image(dirty);
	deconvolved	= destroy_image(deconvolved);
	residual	= destroy_image(residual);
	restored	= destroy_image(restored);
	vt 		= destroy_vis(vt);
	vtpredicted 	= destroy_vis(vtpredicted);
	vt_predictfunction = destroy_vis(vt_predictfunction);
	vt_gt 		= destroy_vis(vt_gt);
	gt 		= destroy_gt(gt);
	free(cindex_predict);
	free(shape);
	free(shape1);
	// end ICAL section

	return 0;

}
