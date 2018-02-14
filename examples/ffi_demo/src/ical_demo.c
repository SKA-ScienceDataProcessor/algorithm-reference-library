#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <math.h>

#include "arlwrap.h"
#include "wrap_support.h"

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
	int *shape = malloc(4*sizeof(int));
	int status;
	int nvis;

	// ICAL section
	int wprojection_planes, i;
	double fstart, fend, fdelta, tstart, tend, tdelta, rmax;
	ARLadvice adv;
	long long int *cindex_predict;
	int cindex_nbytes;
	// end ICAL section

	double cellsize = 0.0005;
	char config_name[] = "LOWBD2-CORE";

	ARLVis *vt;			//Blockvisibility
	ARLVis *vtmodel;
	ARLVis *vtmp;
	ARLVis *vtpredicted;		//Visibility
	ARLVis *vt_predictfunction;	//Blockvisibility

	ARLConf *lowconfig;

	arl_initialize();

	lowconfig = allocate_arlconf_default(config_name);

	// ICAL section
	lowconfig->rmax = 300.0;
	// Overwriting default values for a phasecentre
	lowconfig->pc_ra = 30.0;					// Phasecentre RA
	lowconfig->pc_dec = -60.0;					// Phasecentre Dec
	// Setting values for the frequencies and times
	lowconfig->nfreqs = 5; 						// Number of frequencies
	lowconfig->nchanwidth = 5;					// Number of channel bandwidths
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
	vtpredicted        = allocate_vis_data(lowconfig->npol, nvis);							     //Visibility

	// Allocating cindex array where 8*sizeof(char) is sizeof(python int)
	cindex_nbytes = lowconfig->ntimes * lowconfig->nant * lowconfig->nant * lowconfig->nfreqs * sizeof(long long int);
	
	if (!(cindex_predict = malloc(cindex_nbytes))) {
		free(cindex_predict);
		return 1;
	}
	
	printf("Done...\n");
	// ICAL section	
	// create_blockvisibility()
	printf("Create blockvisibility... ");
	arl_create_blockvisibility(lowconfig, vt);
	printf("Done\n");
	// end ICAL section
	
	// ICAL section
	// adwise_wide_field()
	adv.guard_band_image = 4.0;
	adv.delA=0.02;
	adv.wprojection_planes = 1;
	printf("Calculating wide field parameters... ");
	arl_advise_wide_field(lowconfig, vt, &adv);
	printf("Done.\n");
	printf("Vis_slices = %d,  npixel = %d, cellsize = %e\n", adv.vis_slices, adv.npixel, adv.cellsize);
	cellsize = adv.cellsize;
	// end ICAL section

	// ICAL section
	// create_low_test_image_from_gleam
	helper_get_image_shape_multifreq(lowconfig, adv.cellsize, adv.npixel, shape);
	printf("A shape of the modeled GLEAM image: [ %d, %d, %d, %d]\n", shape[0], shape[1], shape[2], shape[3]);
	Image *gleam_model = allocate_image(shape);
	arl_create_low_test_image_from_gleam(lowconfig, adv.cellsize, adv.npixel, vt->phasecentre, gleam_model);

	// FITS files output
	status = mkdir("results", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	status = export_image_to_fits_c(gleam_model, "results/gleam_model.fits");
	// ICAL section
	// predict_function()
	arl_predict_function(lowconfig, vt, gleam_model, vtpredicted, vt_predictfunction, cindex_predict);

	// convert_visibility_to_blockvisibility()
	arl_convert_visibility_to_blockvisibility(lowconfig, vtpredicted, vt_predictfunction, cindex_predict, vt);

	gleam_model = destroy_image(gleam_model);
	vt = destroy_vis(vt);
	vtpredicted = destroy_vis(vtpredicted);
	vt_predictfunction = destroy_vis(vt_predictfunction);
	free(cindex_predict);
	// end ICAL section

	return 0;

	// Calculate shape of future images, store in 'shape'
	printf("Get image shape... ");
	helper_get_image_shape(lowconfig->freqs, cellsize, shape);
	printf("Done\n");

	printf("Allocate images... ");
	Image *model = allocate_image(shape);
	Image *m31image = allocate_image(shape);
	Image *dirty = allocate_image(shape);
	Image *psf = allocate_image(shape);
	Image *comp = allocate_image(shape);
	Image *residual = allocate_image(shape);
	Image *restored = allocate_image(shape);
	printf("Done\n");

	printf("Create visibility... ");
	arl_create_visibility(lowconfig, vt);
	printf("Done\n");

	/* TODO the vt->phasecentre part should be moved to a separate routine */
	arl_create_test_image(lowconfig->freqs, cellsize, vt->phasecentre, m31image);

	arl_predict_2d(vt, m31image, vtmp);

	vt = destroy_vis(vt);
	vt = vtmp;
	vtmp = NULL;


	arl_create_image_from_visibility(vt, model);

	double *sumwt = malloc(sizeof(double));

	arl_invert_2d(vt, model, false, dirty, sumwt);
	arl_invert_2d(vt, model, true, psf, sumwt);

	arl_deconvolve_cube(dirty, psf, comp, residual);
	arl_restore_cube(comp, psf, residual, restored);

	// FITS files output
	status = mkdir("results", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	status = export_image_to_fits_c(m31image, "results/m31image.fits");
	status = export_image_to_fits_c(dirty, "results/dirty.fits");
	status = export_image_to_fits_c(psf, "results/psf.fits");
	status = export_image_to_fits_c(residual, "results/residual.fits");
	status = export_image_to_fits_c(restored, "results/restored.fits");
	status = export_image_to_fits_c(comp, "results/solution.fits");

	if(status) printf("WARNING: FITSIO status: %d\n", status);

	model = destroy_image(model);
	m31image = destroy_image(m31image);
	dirty = destroy_image(dirty);
	psf = destroy_image(psf);
	residual = destroy_image(residual);
	restored = destroy_image(restored);

	vtmodel = allocate_vis_data(lowconfig->npol, nvis);
	vtmp = allocate_vis_data(lowconfig->npol, nvis);

	arl_create_visibility(lowconfig, vtmodel);

	arl_predict_2d(vtmodel, comp, vtmp);

	vtmodel = destroy_vis(vtmodel);
	vtmp = destroy_vis(vtmp);

	comp = destroy_image(comp);

	return 0;
}
