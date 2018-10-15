#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <math.h>

#include "arlwrap.h"
#include "wrap_support.h"
#include "arlwrap.h"
#include "wrap_support.h"
#include "arlvis_if.h"
#include "ical_pu_routines.h"

/*
 * Verifies that:
 * - vt and vtmp are unique in memory
 * - vt and vtmp have equivalent values
 */
/*
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

*/

int update_weights(double* totalwt,double* totalwt_slice,int* shape1, int i1, int j1){
//	printf("%f\n", totalwt[i1*shape1[1]+j1]);
  totalwt[i1*shape1[1]+j1] += totalwt_slice[i1*shape1[1]+j1];
	//printf("%d %d %f %f\n", i1, j1, totalwt[i1*shape1[1]+j1], totalwt_slice[i1*shape1[1]+j1]);
  return 0;
}

void pu_update_weights(void **buffers, void *cl_args)
{
//  update_weights(((double *)STARPU_VARIABLE_GET_PTR(buffers[0])), (double *)STARPU_VARIABLE_GET_PTR(buffers[1]), *(int *)STARPU_VARIABLE_GET_PTR(buffers[2]), *((int *)STARPU_VARIABLE_GET_PTR(buffers[3])), *((int *)STARPU_VARIABLE_GET_PTR(buffers[4])));
}
void pu_predict_task(void **buffers, void *cl_args)
{
//  printf("\nvisout %d\n",((ARLVis *)STARPU_ARLVIS_GET_CPU_PTR(buffers[4]))->nvis);
//  printf("nvis %d image size %d\n",((ARLVis *)STARPU_VARIABLE_GET_PTR(buffers[1]))->nvis,(*(Image *)STARPU_VARIABLE_GET_PTR(buffers[3])).size );
//  arl_predict_function_oneslice(((ARLConf *)STARPU_VARIABLE_GET_PTR(buffers[0])), (ARLVis *)STARPU_VARIABLE_GET_PTR(buffers[1]), STARPU_VARIABLE_GET_PTR(buffers[2]), ((Image *)STARPU_VARIABLE_GET_PTR(buffers[3])), ((ARLVis *)STARPU_VARIABLE_GET_PTR(buffers[4])));
  arl_predict_function_oneslice(((ARLConf *)STARPU_VARIABLE_GET_PTR(buffers[0])), (ARLVis *)STARPU_VARIABLE_GET_PTR(buffers[1]), STARPU_VARIABLE_GET_PTR(buffers[2]), ((Image *)STARPU_VARIABLE_GET_PTR(buffers[3])), ((ARLVis *)STARPU_ARLVIS_GET_CPU_PTR(buffers[4])));

}

struct starpu_codelet update_weights_cl = {
  .cpu_funcs = { pu_update_weights },
  .name = "pu_update_weights",
  .nbuffers = 5,
  .modes = { STARPU_RW, STARPU_R, STARPU_R, STARPU_R, STARPU_R }
};
struct starpu_codelet predict_task_cl = {
  .cpu_funcs = { pu_predict_task },
  .name = "pu_predict_task",
  .nbuffers = 5,
  .modes = { STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_RW }
};

int main(int argc, char **argv)
{
	int *shape  = malloc(4*sizeof(int));
	int *shape1 = malloc(4*sizeof(int));
	int *c_rows;
	int status;
	int nvis;
  int status_code;
  int i0, i1, j1;  

	int wprojection_planes, i, j, isum, nmajor, first_selfcal;
	double fstart, fend, fdelta, tstart, tend, tdelta, rmax, thresh;
	double *totalwt, *totalwt_slice;
	ARLadvice adv;
	ant_t nb;			//nant and nbases
	long long int *cindex_predict, *cindex_ical;
	long long int *cindex2_ical, *cindex_slice, *cindex_rvis;
	int cindex_nbytes;
	ARLGt *gt;			//GainTable
	ARLGt *gt_ical;			//GainTable (ICAL unrolled)
	bool unrolled_ical = true; // true - unrolled, false - arl.functions::ical()
	bool unrolled_func = true; // true - unrolled invert, predict, etc
	bool unrolled_invert = true; // true - unrolled invert
	bool unrolled_predict = true; // true - unrolled predict

	double cellsize = 0.0005;
	char config_name[] = "LOWBD2-CORE";
	char pol_frame [] = "stokesI";

	ARLVis *vt;			//Blockvisibility
	ARLVis *vt_vis, *vt_rvis;	//Visibility
	ARLVis *vt_bvis, *vt_rbvis;	//Blockvisibility
	ARLVis *vtmodel;
	ARLVis *vtmp;
//	ARLVis *vtpredicted;		//Visibility
	ARLVis *visslice;		//Visibility
	ARLVis *bvisslice;		//Blockvisibility
	ARLVis *vt_predictfunction;	//Blockvisibility
	ARLVis *vt_gt;			//Blockvisibility after gain table applied
	ARLVis *vt_gt_vis;		//Visibility after gain table applied
	ARLVis *vis_ical, *vpred_ical;	//Visibility ICAL temp
	ARLVis *vres_ical;		//Visibility ICAL temp
	ARLVis *bvtmp_ical, *bvtmp2_ical, *bvpred_ical;//Blockvisibility ICAL temp

	ARLConf *lowconfig;

	Image *gleam_model;		//Image (GLEAM model)
	Image *model;			//Image (a model for CLEAN)
	Image *dirty;			//Image (dirty image by invert_function)
	Image *dirty_slice;		//Image (dirty image by invert_function)
	Image *deconvolved;		//Image (ICAL result)
	Image *residual;		//Image (ICAL result)
	Image *restored;		//Image (ICAL result)
	Image *psf_ical;		//Image (ICAL unrolled psf)
	Image *dirty_ical;		//Image (ICAL unrolled dirty)
	Image *cc_ical;			//Image (ICAL unrolled temp restored)
	Image *res_ical;		//Image (ICAL unrolled temp residuals)


	arl_initialize();

	lowconfig = allocate_arlconf_default(config_name);
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

	nvis = (lowconfig->nbases)*(lowconfig->nfreqs)*(lowconfig->ntimes);
	printf("Nvis = %d\n", nvis);
	
	vt 		   = allocate_blockvis_data(lowconfig->nant, lowconfig->nfreqs, lowconfig->npol, lowconfig->ntimes); //Blockvisibility
	vt_vis             = allocate_vis_data(lowconfig->npol, nvis);						     //Visibility
	vt_bvis		   = allocate_blockvis_data(lowconfig->nant, lowconfig->nfreqs, lowconfig->npol, lowconfig->ntimes); //Blockvisibility
	vt_rvis            = allocate_vis_data(lowconfig->npol, nvis);						     //Visibility
	vt_rbvis	   = allocate_blockvis_data(lowconfig->nant, lowconfig->nfreqs, lowconfig->npol, lowconfig->ntimes); //Blockvisibility
	vt_predictfunction = allocate_blockvis_data(lowconfig->nant, lowconfig->nfreqs, lowconfig->npol, lowconfig->ntimes); //Blockvisibility
	vt_gt              = allocate_blockvis_data(lowconfig->nant, lowconfig->nfreqs, lowconfig->npol, lowconfig->ntimes); //Blockvisibility
//	vtpredicted        = allocate_vis_data(lowconfig->npol, nvis);	
	visslice           = allocate_vis_data(lowconfig->npol, nvis);						     //Visibility
	bvisslice          = allocate_blockvis_data(lowconfig->nant, lowconfig->nfreqs, lowconfig->npol, lowconfig->ntimes); //Blockvisibility
	vt_gt_vis          = allocate_vis_data(lowconfig->npol, nvis);						     //Visibility

	// Allocate cindex array where 8*sizeof(char) is sizeof(python int)
	cindex_nbytes = lowconfig->ntimes * lowconfig->nant * lowconfig->nant * lowconfig->nfreqs * sizeof(long long int);
	
	if (!(cindex_predict = malloc(cindex_nbytes))) {
		free(cindex_predict);
		return 1;
	}
	
	if (!(cindex_slice = malloc(cindex_nbytes))) {
		free(cindex_slice);
		return 1;
	}

	if (!(cindex_rvis = malloc(cindex_nbytes))) {
		free(cindex_rvis);
		return 1;
	}

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

///////////////// To be removed later
//	adv.vis_slices = 2;
////////////////

	printf("Done.\n");
	printf("Vis_slices = %d,  npixel = %d, cellsize = %e\n", adv.vis_slices, adv.npixel, adv.cellsize);
	cellsize = adv.cellsize;

	// create_low_test_image_from_gleam, nchan = lowconfig->nfreqs (number of frequencies)
	helper_get_image_shape_multifreq(lowconfig, adv.cellsize, adv.npixel, shape);
	printf("A shape of the modeled GLEAM image: [ %d, %d, %d, %d]\n", shape[0], shape[1], shape[2], shape[3]);
	gleam_model = allocate_image(shape);
	arl_create_low_test_image_from_gleam(lowconfig, adv.cellsize, adv.npixel, vt->phasecentre, gleam_model);

	// FITS file output
	status = mkdir("results", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	status = export_image_to_fits_c(gleam_model, "!results/ical_c_api-gleam_model.fits");

	// For the unrolled functions
	// Allocate c_rows array
	if(!(c_rows = calloc(adv.vis_slices*nvis,sizeof(int)))) {
		free(c_rows);
		return -1;
	}

	// Objects needed for the unrolled functions
	if(unrolled_func) {
		// Fill metadata in the visibility objects
		arl_create_visibility(lowconfig, visslice);
		arl_create_blockvisibility(lowconfig, vt_bvis);
		arl_create_visibility(lowconfig, vt_vis);
		arl_create_blockvisibility(lowconfig, vt_rbvis);
		arl_create_visibility(lowconfig, vt_rvis);

		// Create all rows arrays for each vis_slice, a single rows array for a particular vis_slice can be extracted using
		// a pointer arithmetics, e.g. *(c_rows + i*nvis) where i is vis_slice	
		arl_create_rows(lowconfig, vt, adv.vis_slices, c_rows);
		for( i = 0; i < adv.vis_slices; i++) {
			printf("%d [", i);
			isum = 0;
			for(j = 0; j < nvis; j++) {
				if (j < 10) printf("%d ", *(c_rows + i*nvis +j));
				isum += *(c_rows + i*nvis +j);
				}
			printf("] %d\n", isum);
		}
	}
  status_code = starpu_init(NULL);
  starpu_data_handle_t twt_h;
  starpu_data_handle_t twt_slice_h;
  starpu_data_handle_t shape_h;
  starpu_data_handle_t i_h;
  starpu_data_handle_t j_h;
  starpu_data_handle_t lowconfig_h;
  starpu_data_handle_t visslice_h;
  starpu_data_handle_t bvisslice_h;
  starpu_data_handle_t gleam_model_h;
  starpu_data_handle_t model_h;
  starpu_data_handle_t rvis_h;
  struct starpu_task *predict_task[adv.vis_slices]; 
	starpu_variable_data_register(&lowconfig_h, STARPU_MAIN_RAM,
	(uintptr_t)(lowconfig), sizeof(ARLConf));
//	starpu_variable_data_register(&visslice_h, STARPU_MAIN_RAM,
//	(uintptr_t)(visslice), sizeof(ARLVis ));
//	starpu_variable_data_register(&bvisslice_h, STARPU_MAIN_RAM,
//	(uintptr_t)(bvisslice), sizeof(ARLVis ));
	starpu_variable_data_register(&gleam_model_h, STARPU_MAIN_RAM,
	(uintptr_t)(gleam_model), sizeof(Image));
	starpu_arlvis_data_register(&visslice_h, STARPU_MAIN_RAM,
	visslice, &visslice->nvis,&visslice->npol,(void *)&visslice->data,(char *)&visslice->phasecentre);
	starpu_arlvis_data_register(&bvisslice_h, STARPU_MAIN_RAM,
	bvisslice, &bvisslice->nvis,&bvisslice->npol,(void *)&bvisslice->data,(char *)&bvisslice->phasecentre);
	starpu_arlvis_data_register(&rvis_h, STARPU_MAIN_RAM,
	vt_rvis, &vt_rvis->nvis,&vt_rvis->npol,(void *)&vt_rvis->data,(char *)&vt_rvis->phasecentre);
//	starpu_variable_data_register(&rvis_h, STARPU_MAIN_RAM,
//	(uintptr_t)(vt_rvis), sizeof(ARLVis));


 
	// predict_function()
	if(unrolled_predict) {
	// convert_blockvisibility_to_visibility
		arl_convert_blockvisibility_to_visibility(lowconfig, vt, vt_vis, cindex_predict, vt_bvis);
		for( i = 0; i < adv.vis_slices; i++) {
			printf("Predict_function vt/vt_vis, vis slice %d, ", i);
			// A version of create_vis_from_rows using vis input - fast
			arl_create_vis_from_rows_vis(lowconfig, vt_vis, cindex_predict, vt_bvis, visslice, cindex_slice, bvisslice, (c_rows + i*nvis));
			arl_set_visibility_data_to_zero(lowconfig, visslice);			
			// copy_visibility (vis, zero=True)
			arl_copy_visibility(lowconfig, visslice, vt_rvis, 0);		
  	  predict_task[i] = starpu_task_create();
      predict_task[i]->cl = &predict_task_cl;
      predict_task[i]->handles[0] = lowconfig_h;
      predict_task[i]->modes[0] = STARPU_R;
      predict_task[i]->handles[1] = visslice_h;
      predict_task[i]->modes[1] = STARPU_R;
      predict_task[i]->handles[2] = bvisslice_h;
      predict_task[i]->modes[2] = STARPU_R;
      predict_task[i]->handles[3] = gleam_model_h;
      predict_task[i]->modes[3] = STARPU_R;
      predict_task[i]->handles[4] = rvis_h;
      predict_task[i]->modes[4] = STARPU_RW;
      starpu_task_submit(predict_task[i]);
      starpu_task_wait_for_all();
 	//arl_set_visibility_data_to_zero(lowconfig, vt_rvis);			
//			arl_predict_function_oneslice(lowconfig, visslice, bvisslice, gleam_model, vt_rvis);
			printf(" nvis = %d %d %d\n", visslice->nvis, vt_rvis->nvis, nvis);
			arl_add_to_visibility_data_slice(lowconfig, vt_vis, vt_rvis, (c_rows + i*nvis));
		}
    starpu_task_wait_for_all();

	// convert_visibility_to_blockvisibility()
		arl_convert_visibility_to_blockvisibility(lowconfig, vt_vis, vt_bvis, cindex_predict, vt);
		
			
	} else {	
		arl_predict_function_blockvis(lowconfig, vt, gleam_model, adv.vis_slices);
	}


	// create_gaintable_from_blockvisibility()
	arl_create_gaintable_from_blockvisibility(lowconfig, vt, gt);

	// simulate_gaintable()
	arl_simulate_gaintable(lowconfig, gt);

	// apply_gaintable()
	arl_apply_gaintable(lowconfig, vt, gt, vt_gt, 1);

	// Create a "model" image with nchan = 1
	for(i = 0; i< 4; i++) {
		shape1[i] = shape[i];
		}
	shape1[0] = 1;
	model = allocate_image(shape1);

	// create_image_from_blockvisibility() (temp image in the inversion loop)
	arl_create_image_from_blockvisibility(lowconfig, vt, adv.cellsize, adv.npixel, vt->phasecentre, model);

	// create a "dirty" image with nchan = 1 (resulting image in the invert loop)
	dirty = allocate_image(shape1);
	arl_create_image_from_blockvisibility(lowconfig, vt, adv.cellsize, adv.npixel, vt->phasecentre, dirty);

	// create a "dirty_slice" image with nchan = 1 (temp image in the invert loops)
	dirty_slice = allocate_image(shape1);
	arl_create_image_from_blockvisibility(lowconfig, vt, adv.cellsize, adv.npixel, vt->phasecentre, dirty_slice);

	// convert_blockvisibility_to_visibility
	arl_convert_blockvisibility_to_visibility(lowconfig, vt_gt, vt_gt_vis, cindex_predict, vt);

	// invert_function()

	// For the unrolled invert functions
	// Allocate total weight arrays for the normalization and set them to zero
		
	if(!(totalwt = calloc(shape1[0]*shape1[1],sizeof(double)))) {
		free(totalwt);
		return -1;
	}

	if(!(totalwt_slice = calloc(shape1[0]*shape1[1],sizeof(double)))) {
		free(totalwt_slice);
		return -1;
	}
	
//	return 0;
  struct starpu_task *update_weights_task[shape1[0]*shape1[1]];
// An unrolled loop for the invert_serial() function
	if(unrolled_invert) {
// A loop over the visibility slices with precompiled rows = *(c_rows + i*nvis) for the invert function
		
		for( i = 0; i < adv.vis_slices; i++) {
			printf("Invert_function dirty, vis slice %d, ", i);
//			arl_set_visibility_data_to_zero(lowconfig, visslice);			

			// A version of create_vis_from_rows using blockvis input, requires convert_blockvisibility_to_visibility() inside every time - slow
//			arl_create_vis_from_rows_blockvis(lowconfig, vt_gt, visslice, cindex_slice, bvisslice, (c_rows + i*nvis));

			// A version of create_vis_from_rows using vis input - fast
			arl_create_vis_from_rows_vis(lowconfig, vt_gt_vis, cindex_predict, vt, visslice, cindex_slice, bvisslice, (c_rows + i*nvis));

			printf(" nvis = %d\n", visslice->nvis);
			arl_invert_function_oneslice(lowconfig, visslice, bvisslice, model, adv.vis_slices, dirty_slice, totalwt_slice, 1); // dopsf = false
			// Update weights			
			printf("Total weights:\n");
//    	starpu_variable_data_register(&twt_h, STARPU_MAIN_RAM,
//			(uintptr_t)(&totalwt), sizeof(double*));
//    	starpu_variable_data_register(&twt_slice_h, STARPU_MAIN_RAM,
//			(uintptr_t)(&totalwt_slice), sizeof(double));
//    	starpu_variable_data_register(&shape_h, STARPU_MAIN_RAM,
//			(uintptr_t)&shape1, sizeof(int));
//    	starpu_variable_data_register(&i_h, STARPU_MAIN_RAM,
//			(uintptr_t)&i1, sizeof(int));
//    	starpu_variable_data_register(&j_h, STARPU_MAIN_RAM,
//			(uintptr_t)&j1, sizeof(int));
			for (i1 = 0; i1 < shape1[0]; i1++) {
				for (j1 = 0; j1 < shape1[1]; j1++) {
            int c = i1*shape1[0]+j1;
//            update_weights_task[c] = starpu_task_create();
//            update_weights_task[c]->cl = &update_weights_cl;
//            update_weights_task[c]->handles[0] = twt_h;
//            update_weights_task[c]->modes[0] = STARPU_RW;
//            update_weights_task[c]->handles[1] = twt_slice_h;
//            update_weights_task[c]->modes[1] = STARPU_R;
//            update_weights_task[c]->handles[2] = shape_h;
//            update_weights_task[c]->modes[2] = STARPU_R;
//            update_weights_task[c]->handles[3] = i_h;
//            update_weights_task[c]->modes[3] = STARPU_R;
//            update_weights_task[c]->handles[4] = j_h;
//            update_weights_task[c]->modes[4] = STARPU_R;
//            status_code = starpu_task_submit(update_weights_task[c]);
//            update_weights(&totalwt, totalwt_slice, shape, i1, j1);
					totalwt[i1*shape1[1]+j1] += totalwt_slice[i1*shape1[1]+j1];
//					printf("%d %d %f %f\n", i1, j1, totalwt[i1*shape1[1]+j1], totalwt_slice[i1*shape1[1]+j1]);
				}
        starpu_task_wait_for_all();
			}
      
			// Update the image
			arl_add_to_model(dirty, dirty_slice);
		}
// Normalize the resulting image
		arl_normalize_sumwt(dirty, totalwt);


	} else {			
	// Original function
		arl_invert_function_blockvis(lowconfig, vt_gt, model, adv.vis_slices, dirty);
	}
  
	// FITS file output
	status = export_image_to_fits_c(dirty, "!results/ical_c_api-dirty.fits");
//	return 0;
	// ical() - serial version
	// create images with nchan = 1
	deconvolved = allocate_image(shape1);
	residual    = allocate_image(shape1);
	restored    = allocate_image(shape1);

	if(unrolled_ical){
	// The same values as hard-coded in arlwrap.py calls
		nmajor = 5; 
		thresh = 0.1;
		first_selfcal = 10;
		printf("ical: Performing %d major cycles\n", nmajor);
	// Allocate temp objects
		vis_ical  = allocate_vis_data(lowconfig->npol, nvis);							     //ICAL Visibility object (ical.vis)
		vpred_ical  = allocate_vis_data(lowconfig->npol, nvis);							     //ICAL Visibility object (ical.vispred)
		vres_ical  = allocate_vis_data(lowconfig->npol, nvis);							     //ICAL Visibility object (ical.visres)
		bvtmp_ical = allocate_blockvis_data(lowconfig->nant, lowconfig->nfreqs, lowconfig->npol, lowconfig->ntimes); //Blockvisibility vis_ical.blockvis (ical.vis.blockvis)
		bvtmp2_ical = allocate_blockvis_data(lowconfig->nant, lowconfig->nfreqs, lowconfig->npol, lowconfig->ntimes); //Blockvisibility vpred_ical.blockvis (ical.vispred.blockvis)
		bvpred_ical = allocate_blockvis_data(lowconfig->nant, lowconfig->nfreqs, lowconfig->npol, lowconfig->ntimes); //Blockvisibility (ical.block_vispred)
		psf_ical    = allocate_image(shape1);									// Image PSF
		arl_create_image_from_blockvisibility(lowconfig, vt, adv.cellsize, adv.npixel, vt->phasecentre, psf_ical); // Fill metadata (phasecentre etc)
		dirty_ical    = allocate_image(shape1);									// Image dirty (CLEAN loop)
		arl_create_image_from_blockvisibility(lowconfig, vt, adv.cellsize, adv.npixel, vt->phasecentre, dirty_ical); // Fill metadata (phasecentre etc)

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
			starpu_variable_data_register(&lowconfig_h, STARPU_MAIN_RAM,
	(uintptr_t)(lowconfig), sizeof(ARLConf));
//	starpu_variable_data_register(&visslice_h, STARPU_MAIN_RAM,
//	(uintptr_t)(visslice), sizeof(ARLVis ));
//	starpu_variable_data_register(&bvisslice_h, STARPU_MAIN_RAM,
//	(uintptr_t)(bvisslice), sizeof(ARLVis ));
	starpu_variable_data_register(&gleam_model_h, STARPU_MAIN_RAM,
	(uintptr_t)(gleam_model), sizeof(Image));
	starpu_arlvis_data_register(&visslice_h, STARPU_MAIN_RAM,
	visslice, &visslice->nvis,&visslice->npol,(void *)&visslice->data,(char *)&visslice->phasecentre);
	starpu_arlvis_data_register(&bvisslice_h, STARPU_MAIN_RAM,
	bvisslice, &bvisslice->nvis,&bvisslice->npol,(void *)&bvisslice->data,(char *)&bvisslice->phasecentre);
	starpu_arlvis_data_register(&rvis_h, STARPU_MAIN_RAM,
	vt_rvis, &vt_rvis->nvis,&vt_rvis->npol,(void *)&vt_rvis->data,(char *)&vt_rvis->phasecentre);
//	starpu_variable_data_register(&rvis_h, STARPU_MAIN_RAM,
//	(uintptr_t)(vt_rvis), sizeof(ARLVis));

if(unrolled_predict) {

    	starpu_variable_data_register(&model_h, STARPU_MAIN_RAM,
    	(uintptr_t)(model), sizeof(Image));
			for( i = 0; i < adv.vis_slices; i++) {
				printf("Predict_function vpred_ical, Vis slice %d, ", i);
				// A version of create_vis_from_rows using vis input - fast
				arl_create_vis_from_rows_vis(lowconfig, vpred_ical, cindex2_ical, bvtmp2_ical, visslice, cindex_slice, bvisslice, (c_rows + i*nvis));
				arl_set_visibility_data_to_zero(lowconfig, visslice);			
				// copy_visibility (vis, zero=True)
				arl_copy_visibility(lowconfig, visslice, vt_rvis, 0);		
				//arl_set_visibility_data_to_zero(lowconfig, vt_rvis);			
				predict_task[i] = starpu_task_create();
        predict_task[i]->cl = &predict_task_cl;
        predict_task[i]->handles[0] = lowconfig_h;
        predict_task[i]->modes[0] = STARPU_R;
        predict_task[i]->handles[1] = visslice_h;
        predict_task[i]->modes[1] = STARPU_R;
        predict_task[i]->handles[2] = bvisslice_h;
        predict_task[i]->modes[2] = STARPU_R;
        predict_task[i]->handles[3] = model_h;
        predict_task[i]->modes[3] = STARPU_R;
        predict_task[i]->handles[4] = rvis_h;
        predict_task[i]->modes[4] = STARPU_RW;

       starpu_task_submit(predict_task[i]);
       starpu_task_wait_for_all();
//				arl_predict_function_oneslice(lowconfig, visslice, bvisslice, model, vt_rvis);
				printf(" nvis = %d %d %d\n", visslice->nvis, vt_rvis->nvis, nvis);
				arl_add_to_visibility_data_slice(lowconfig, vpred_ical, vt_rvis, (c_rows + i*nvis));
			}
		} else {
			arl_predict_function_ical(lowconfig, vpred_ical, model, bvtmp2_ical, cindex2_ical, adv.vis_slices);
		}
	// convert_visibility_to_blockvisibility()
		arl_convert_visibility_to_blockvisibility(lowconfig, vpred_ical, bvtmp2_ical, cindex2_ical, bvpred_ical);
//return 0;
	// Subtract visibility data to find residuals
        // vres = vtmp - vpred : 0 = add, 1 = subtract, 2 = mult, 3 = divide, else sets to zero
		arl_manipulate_visibility_data(lowconfig, vis_ical, vpred_ical, vres_ical, 1); 

		if(unrolled_invert) {
			// To be replaced with an unrolled version
			// dirty_ical should be set to zero when created
			// Reset total weights (to be replaced with memcpy?)
			for (i1 = 0; i1 < shape1[0]; i1++) {
				for (j1 = 0; j1 < shape1[1]; j1++) {
					totalwt[i1*shape1[1]+j1] = 0.0;
				}
			}
			// totalwt_slice can be re-used
			// dirty_slice and visslice can be re-used

			for( i = 0; i < adv.vis_slices; i++) {
				printf("arl_invert_function_ical dirty, vis slice %d, ", i);

			// A version of create_vis_from_rows using vis input - fast
				arl_create_vis_from_rows_vis(lowconfig, vres_ical, cindex_ical, bvtmp_ical, visslice, cindex_slice, bvisslice, (c_rows + i*nvis));

				printf(" nvis = %d\n", visslice->nvis);
				arl_invert_function_oneslice(lowconfig, visslice, bvisslice, model, adv.vis_slices, dirty_slice, totalwt_slice, 1); // dopsf = false
				// Update weights			
				printf("Total weights:\n");
				for (i1 = 0; i1 < shape1[0]; i1++) {
					for (j1 = 0; j1 < shape1[1]; j1++) {
						totalwt[i1*shape1[1]+j1] += totalwt_slice[i1*shape1[1]+j1];
						printf("%d %d %f %f\n", i1, j1, totalwt[i1*shape1[1]+j1], totalwt_slice[i1*shape1[1]+j1]);
					}
				}
				// Update the image
				arl_add_to_model(dirty_ical, dirty_slice);
			}
			// Normalize the resulting image
			arl_normalize_sumwt(dirty_ical, totalwt);

			//arl_invert_function_ical(lowconfig, vres_ical, model, adv.vis_slices, dirty_ical);
		} else {
	// arl_invert_function_ical() (extra parameters in **kwargs -TBS later)
			arl_invert_function_ical(lowconfig, vres_ical, model, adv.vis_slices, dirty_ical);
		}

		// This loop can be merged with the upper one since they work on the same vis data constructing a dirty image and a PSF
		if(unrolled_invert) {
			// To be replaced with an unrolled version
			// dirty_ical should be set to zero when created
			// Reset total weights (to be replaced with memcpy?)
			for (i1 = 0; i1 < shape1[0]; i1++) {
				for (j1 = 0; j1 < shape1[1]; j1++) {
					totalwt[i1*shape1[1]+j1] = 0.0;
				}
			}
			// totalwt_slice can be re-used
			// dirty_slice and visslice can be re-used

			for( i = 0; i < adv.vis_slices; i++) {
				printf("arl_invert_function_ical psf, vis slice %d, ", i);

			// A version of create_vis_from_rows using vis input - fast
				arl_create_vis_from_rows_vis(lowconfig, vres_ical, cindex_ical, bvtmp_ical, visslice, cindex_slice, bvisslice, (c_rows + i*nvis));

				printf(" nvis = %d\n", visslice->nvis);
				arl_invert_function_oneslice(lowconfig, visslice, bvisslice, model, adv.vis_slices, dirty_slice, totalwt_slice, 0); // dopsf = true
				// Update weights			
				printf("Total weights:\n");
				for (i1 = 0; i1 < shape1[0]; i1++) {
					for (j1 = 0; j1 < shape1[1]; j1++) {
            int c = i1*shape1[0]+j1;
//            update_weights_task[c] = starpu_task_create();
//            update_weights_task[c]->cl = &update_weights_cl;
//            update_weights_task[c]->handles[0] = twt_h;
//            update_weights_task[c]->modes[0] = STARPU_RW;
//            update_weights_task[c]->handles[1] = twt_slice_h;
//            update_weights_task[c]->modes[1] = STARPU_R;
//            update_weights_task[c]->handles[2] = shape_h;
//            update_weights_task[c]->modes[2] = STARPU_R;
//            update_weights_task[c]->handles[3] = i_h;
//            update_weights_task[c]->modes[3] = STARPU_R;
//            update_weights_task[c]->handles[4] = j_h;
//            status_code = starpu_task_submit(update_weights_task[c]);
						totalwt[i1*shape1[1]+j1] += totalwt_slice[i1*shape1[1]+j1];
						printf("%d %d %f %f\n", i1, j1, totalwt[i1*shape1[1]+j1], totalwt_slice[i1*shape1[1]+j1]);
					}
				}
				// Update the image
				arl_add_to_model(psf_ical, dirty_slice);
			}
			// Normalize the resulting image
			arl_normalize_sumwt(psf_ical, totalwt);
		} else {

		// arl_invert_function_psf() (extra parameters in **kwargs -TBS later)
		arl_invert_function_psf(lowconfig, vres_ical, model, adv.vis_slices, psf_ical);
		}
			starpu_variable_data_register(&lowconfig_h, STARPU_MAIN_RAM,
	(uintptr_t)(lowconfig), sizeof(ARLConf));
//	starpu_variable_data_register(&visslice_h, STARPU_MAIN_RAM,
//	(uintptr_t)(visslice), sizeof(ARLVis ));
//	starpu_variable_data_register(&bvisslice_h, STARPU_MAIN_RAM,
//	(uintptr_t)(bvisslice), sizeof(ARLVis ));
	starpu_variable_data_register(&gleam_model_h, STARPU_MAIN_RAM,
	(uintptr_t)(gleam_model), sizeof(Image));
	starpu_arlvis_data_register(&visslice_h, STARPU_MAIN_RAM,
	visslice, &visslice->nvis,&visslice->npol,(void *)&visslice->data,(char *)&visslice->phasecentre);
	starpu_arlvis_data_register(&bvisslice_h, STARPU_MAIN_RAM,
	bvisslice, &bvisslice->nvis,&bvisslice->npol,(void *)&bvisslice->data,(char *)&bvisslice->phasecentre);
	starpu_arlvis_data_register(&rvis_h, STARPU_MAIN_RAM,
	vt_rvis, &vt_rvis->nvis,&vt_rvis->npol,(void *)&vt_rvis->data,(char *)&vt_rvis->phasecentre);
//	starpu_variable_data_register(&rvis_h, STARPU_MAIN_RAM,
//	(uintptr_t)(vt_rvis), sizeof(ARLVis));


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
			if(unrolled_predict) {
				for( i0 = 0; i0 < adv.vis_slices; i0++) {
					printf("Predict_function vpred_ical (in a CLEAN loop), Vis slice %d, ", i0);
					// A version of create_vis_from_rows using vis input - fast
					arl_create_vis_from_rows_vis(lowconfig, vpred_ical, cindex2_ical, bvtmp2_ical, visslice, cindex_slice, bvisslice, (c_rows + i0*nvis));
					arl_set_visibility_data_to_zero(lowconfig, visslice);			
					// copy_visibility (vis, zero=True)
					arl_copy_visibility(lowconfig, visslice, vt_rvis, 0);		
				predict_task[i0] = starpu_task_create();
        predict_task[i0]->cl = &predict_task_cl;
        predict_task[i0]->handles[0] = lowconfig_h;
        predict_task[i0]->modes[0] = STARPU_R;
        predict_task[i0]->handles[1] = visslice_h;
        predict_task[i0]->modes[1] = STARPU_R;
        predict_task[i0]->handles[2] = bvisslice_h;
        predict_task[i0]->modes[2] = STARPU_R;
        predict_task[i0]->handles[3] = model_h;
        predict_task[i0]->modes[3] = STARPU_R;
        predict_task[i0]->handles[4] = rvis_h;
        predict_task[i0]->modes[4] = STARPU_RW;

        starpu_task_submit(predict_task[i0]);
        starpu_task_wait_for_all();
//arl_set_visibility_data_to_zero(lowconfig, vt_rvis);			
//					arl_predict_function_oneslice(lowconfig, visslice, bvisslice, model, vt_rvis);
					printf(" nvis = %d %d %d\n", visslice->nvis, vt_rvis->nvis, nvis);
					arl_add_to_visibility_data_slice(lowconfig, vpred_ical, vt_rvis, (c_rows + i0*nvis));
				}
        starpu_task_wait_for_all();
			} else {

				arl_predict_function_ical(lowconfig, vpred_ical, model, bvtmp2_ical, cindex2_ical, adv.vis_slices);
			}
		// if doselfcal (currently not working, to be replaced with calibrate_function)
/*			if(i >= first_selfcal) {
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
*/			
        	// vres = vtmp - vpred : 0 = add, 1 = subtract, 2 = mult, 3 = divide, else sets to zero
			arl_manipulate_visibility_data(lowconfig, vis_ical, vpred_ical, vres_ical, 1); 

		// arl_invert_function_ical() (extra parameters in **kwargs -TBS later)
			if(unrolled_invert) {
			// To be replaced with an unrolled version
			// dirty_ical data should be set to zero 
				memset(dirty_ical->data, 0, sizeof(double) * dirty_ical->size);
			// Reset total weights (to be replaced with memcpy?)
				for (i1 = 0; i1 < shape1[0]; i1++) {
					for (j1 = 0; j1 < shape1[1]; j1++) {
						totalwt[i1*shape1[1]+j1] = 0.0;
					}
				}
				// totalwt_slice can be re-used
				// dirty_slice and visslice can be re-used
		
				for( i0 = 0; i0 < adv.vis_slices; i0++) {
					printf("arl_invert_function_ical dirty (in a CLEAN loop), vis slice %d, ", i0);

				// A version of create_vis_from_rows using vis input - fast
					arl_create_vis_from_rows_vis(lowconfig, vres_ical, cindex_ical, bvtmp_ical, visslice, cindex_slice, bvisslice, (c_rows + i0*nvis));
	
					printf(" nvis = %d\n", visslice->nvis);
					arl_invert_function_oneslice(lowconfig, visslice, bvisslice, model, adv.vis_slices, dirty_slice, totalwt_slice, 1); // dopsf = false
					// Update weights			
					printf("Total weights:\n");
					for (i1 = 0; i1 < shape1[0]; i1++) {
						for (j1 = 0; j1 < shape1[1]; j1++) {
							totalwt[i1*shape1[1]+j1] += totalwt_slice[i1*shape1[1]+j1];
							printf("%d %d %f %f\n", i1, j1, totalwt[i1*shape1[1]+j1], totalwt_slice[i1*shape1[1]+j1]);
						}
					}
					// Update the image
					arl_add_to_model(dirty_ical, dirty_slice);
				}
				// Normalize the resulting image
				arl_normalize_sumwt(dirty_ical, totalwt);

				//arl_invert_function_ical(lowconfig, vres_ical, model, adv.vis_slices, dirty_ical);
			} else {
				arl_invert_function_ical(lowconfig, vres_ical, model, adv.vis_slices, dirty_ical);
			}
		// ToDo - loop break on threshold

			printf("ical: End of major cycle %d\n", i);

		}
		printf("ical: End of major cycles\n");
		arl_restore_cube_ical(model, psf_ical, dirty_ical, restored);

	// FITS file output, unrolled version of the files
		status = export_image_to_fits_c(model, 		"!results/ical_c_api_u_deconvolved.fits"); 
		status = export_image_to_fits_c(dirty_ical,	"!results/ical_c_api_u_residual.fits");
		status = export_image_to_fits_c(restored, 	"!results/ical_c_api_u_restored.fits");
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

//		model = destroy_image(model);
//		model = allocate_image(shape1);
//		arl_create_image_from_blockvisibility(lowconfig, vt_gt, adv.cellsize, adv.npixel, vt_gt->phasecentre, model);

		arl_ical(lowconfig, vt_gt, model, adv.vis_slices, deconvolved, residual, restored);
	// FITS file output, normal versions of the files
		status = export_image_to_fits_c(deconvolved, 	"!results/ical_c_api_deconvolved.fits");
		status = export_image_to_fits_c(residual, 	"!results/ical_c_api_residual.fits");
		status = export_image_to_fits_c(restored, 	"!results/ical_c_api_restored.fits");
	}


	// Cleaning up
	gleam_model 	= destroy_image(gleam_model);
	model		= destroy_image(model);
	dirty		= destroy_image(dirty);
	dirty_slice	= destroy_image(dirty_slice);
	deconvolved	= destroy_image(deconvolved);
	residual	= destroy_image(residual);
	restored	= destroy_image(restored);
	vt 		= destroy_vis(vt);
	vt_vis 		= destroy_vis(vt_vis);
	vt_bvis 	= destroy_vis(vt_bvis);
	vt_rvis 	= destroy_vis(vt_rvis);
	vt_rbvis 	= destroy_vis(vt_rbvis);

//	vtpredicted 	= destroy_vis(vtpredicted);
	vt_predictfunction = destroy_vis(vt_predictfunction);
	vt_gt 		= destroy_vis(vt_gt);
	vt_gt_vis 	= destroy_vis(vt_gt_vis);
	visslice	= destroy_vis(visslice);
	bvisslice	= destroy_vis(bvisslice);
	gt 		= destroy_gt(gt);
	free(cindex_predict);
	free(cindex_slice);
	free(shape);
	free(shape1);
	free(c_rows); 
// Free total weight arrays
	free(totalwt);
	free(totalwt_slice);
//	starpu_data_unregister(&lowconfig_h);
//	starpu_data_unregister(&visslice_h);
//	starpu_data_unregister(&bvisslice_h);
//	starpu_data_unregister(&gleam_model_h);
//	starpu_data_unregister(&model_h);
//	starpu_data_unregister(&rvis_h);

  starpu_shutdown();

	return 0;

}
