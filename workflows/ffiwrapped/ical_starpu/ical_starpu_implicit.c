/* 
 * ical_starpu.c
 *
 * Implements a basic ICAL pipeline using StarPU and the ARL C Wrappers.
 *
 * Adapted from the TIMG StarPU demo
 *
 * Author: Chris Hadjigeorgiou <ch741@cam.ac.uk>
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <math.h>
#include <starpu.h>

#include "arlwrap.h"
#include "wrap_support.h"
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
//  starpu_data_handle_t _h[5];
//  SVDR(_h, 0, , sizeof());
//  SVDR(_h, , , sizeof());
//  SVDR(_h, , , sizeof());
//  SVDR(_h, , , sizeof());
//  SVDR(_h, , , sizeof());
//  
//  struct starpu_task *_task = create_task(_cl, _h);
//  starpu_task_declare_deps_array(_task, 1, &_task);
//  starpu_task_submit(_task);

int main(int argc, char **argv)
{
	int *shape = malloc(4*sizeof(int));
	int *shape1 = malloc(4*sizeof(int));
	int status;
	int nvis;
  int status_code;
  int inv_in = 1;
  int zero_in = 0;
  int op = 1;

	int wprojection_planes, i, nmajor, first_selfcal;
	double fstart, fend, fdelta, tstart, tend, tdelta, rmax, thresh;
  ARLadvice adv;
//	ARLadvice* adv = allocate_arladv_default();
//  ARLadv->ce adv->
//  adv->vis_slices= 0;
//  adv->npixel = 0;
//  adv->cellsize = 0;
	adv.guard_band_image = 4.0;
	adv.delA=0.02;
	adv.wprojection_planes = 1;
    // malloc(sizeof(*adv->);//
	ant_t nb;			//nant and nbases
	long long int *cindex_predict, *cindex_ical, *cindex2_ical;
	int cindex_nbytes;
	ARLGt *gt;			//GainTable
	ARLGt *gt_ical;			//GainTable (ICAL unrolled)
	bool unrolled = true; // true - unrolled, false - arl.functions::ical()

  int npixel;// = malloc(sizeof(int));
	double cellsize;// = malloc(sizeof(double));
  
  cellsize = 0.0005;
	char config_name[] = "LOWBD2-CORE";
	char pol_frame [] = "stokesI";
  char *phasecentre;

	ARLVis *vt;			//Blockvisibility
	ARLVis *vtmodel;
	ARLVis *vtmp;
//	ARLVis *vtpredicted;		//Visibility
	ARLVis *vt_predictfunction;	//Blockvisibility
	ARLVis *vt_gt;			//Blockvisibility after gain table applied
	ARLVis *vis_ical, *vpred_ical;	//Visibility ICAL temp
	ARLVis *vres_ical;		//Visibility ICAL temp
	ARLVis *bvtmp_ical, *bvtmp2_ical, *bvpred_ical;//Blockvisibility ICAL temp

//	ARLConf *lowconfig;

	Image *model;			//Image (a model for CLEAN)
	Image *dirty;			//Image (dirty image by invert_function)
	Image *deconvolved;		//Image (ICAL result)
	Image *residual;		//Image (ICAL result)
	Image *restored;		//Image (ICAL result)
	Image *psf_ical;		//Image (ICAL unrolled psf)
	Image *dirty_ical;		//Image (ICAL unrolled dirty)
	Image *cc_ical;			//Image (ICAL unrolled temp restored)
	Image *res_ical;		//Image (ICAL unrolled temp residuals)

  status_code = starpu_init(NULL);

	arl_initialize();

//  starpu_malloc((void *)shape,sizeof(int));
// starpu_malloc((void *)shape1,sizeof(int));
	ARLConf *lowconfig = allocate_arlconf_default(config_name);
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
	// Overwriting default frequency list
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
	vt_predictfunction = allocate_blockvis_data(lowconfig->nant, lowconfig->nfreqs, lowconfig->npol, lowconfig->ntimes); //Blockvisibility
	vt_gt              = allocate_blockvis_data(lowconfig->nant, lowconfig->nfreqs, lowconfig->npol, lowconfig->ntimes); //Blockvisibility
//	vtpredicted        = allocate_vis_data(lowconfig->npol, nvis);							     //Visibility

	// Allocate cindex array where 8*sizeof(char) is sizeof(python int)
	cindex_nbytes = lowconfig->ntimes * lowconfig->nant * lowconfig->nant * lowconfig->nfreqs * sizeof(long long int);
	
	if (!(cindex_predict = malloc(cindex_nbytes))) {
		free(cindex_predict);
		return 1;
	}
	
	// create_blockvisibility()
	printf("Create blockvisibility... ");
  starpu_data_handle_t lowconfig_h;
  starpu_data_handle_t vt_h;
	starpu_variable_data_register(&lowconfig_h, STARPU_MAIN_RAM,
      (uintptr_t)lowconfig, sizeof(ARLConf));
	starpu_variable_data_register(&vt_h, STARPU_MAIN_RAM,
			(uintptr_t)vt, sizeof(ARLVis));

	struct starpu_task *blockvis_task = starpu_task_create();
  blockvis_task->cl = &create_blockvisibility_cl;
  blockvis_task->handles[0] = lowconfig_h;
  blockvis_task->modes[0] = STARPU_RW;
  blockvis_task->handles[1] = vt_h;
  blockvis_task->modes[1] = STARPU_W;
	status_code = starpu_task_submit(blockvis_task);
  printf("Done.\n");
//	arl_create_blockvisibility(lowconfig, vt);
	printf("Nrec = %d\n", lowconfig->nrec);
	// Allocating gaintable data
	gt = allocate_gt_data(lowconfig->nant, lowconfig->nfreqs, lowconfig->nrec, lowconfig->ntimes);
	// adwise_wide_field()
	printf("Calculating wide field parameters... ");
  starpu_data_handle_t adv_h;
	starpu_variable_data_register(&adv_h, STARPU_MAIN_RAM,
			(uintptr_t)&adv, sizeof(ARLadvice*));

  struct starpu_task *advise_wide_field_task = starpu_task_create();
  
  advise_wide_field_task->cl = &advise_wide_field_cl;
  advise_wide_field_task->handles[0] = lowconfig_h;
  advise_wide_field_task->modes[0] = STARPU_R;
  advise_wide_field_task->handles[1] = vt_h;
  advise_wide_field_task->modes[1] = STARPU_R;
  advise_wide_field_task->handles[2] = adv_h;
  advise_wide_field_task->modes[2] = STARPU_RW;
  advise_wide_field_task->synchronous = 1 ;
//  struct starpu_task *advise_wide_field_task = create_task(&advise_wide_field_cl, advise_wide_field_h);

  status_code = starpu_task_submit(advise_wide_field_task);

  cellsize = (adv.cellsize);
  npixel = (adv.npixel);
//  arl_advise_wide_field(lowconfig, vt, &adv);
//	printf("Done.\n");
	printf("Vis_slices = %d,  npixel = %d, cellsize = %e\n", adv.vis_slices, adv.npixel, adv.cellsize);
	// create_low_test_image_from_gleam, nchan = lowconfig->nfreqs (number of frequencies)
	//
//	helper_get_image_shape_multifreq(lowconfig, adv.cellsize, adv.npixel, shape);
//  starpu_data_handle_t helper_get_image_shape_multifreq_h[4];
//  SVDR(helper_get_image_shape_multifreq_h, 0, lowconfig, sizeof(ARLConf));	
//  SVDR(helper_get_image_shape_multifreq_h, 1, &(adv->cellsize), sizeof(double));	
//  SVDR(helper_get_image_shape_multifreq_h, 2, &(adv->npixel), sizeof(int));	
//  SVDR(helper_get_image_shape_multifreq_h, 3, &shape, sizeof(shape[0]));	
  starpu_data_handle_t cellsize_h;  
  starpu_data_handle_t npixel_h;  
  starpu_data_handle_t shape_h;  
	starpu_variable_data_register(&adv_h, STARPU_MAIN_RAM,
  	(uintptr_t)&(cellsize), sizeof(double));
	starpu_variable_data_register(&npixel_h, STARPU_MAIN_RAM,
			(uintptr_t)&(adv.npixel), sizeof(int));
	starpu_variable_data_register(&shape_h, STARPU_MAIN_RAM,
			(uintptr_t)shape, 4*sizeof(int));

  struct starpu_task *helper_get_image_shape_multifreq_task = starpu_task_create();
  helper_get_image_shape_multifreq_task->cl = &helper_get_image_shape_multifreq_cl;
  helper_get_image_shape_multifreq_task->handles[0] = lowconfig_h;
  helper_get_image_shape_multifreq_task->modes[0] = STARPU_R;
  helper_get_image_shape_multifreq_task->handles[1] = adv_h;
  helper_get_image_shape_multifreq_task->modes[1] = STARPU_R;
  helper_get_image_shape_multifreq_task->handles[2] = npixel_h;
  helper_get_image_shape_multifreq_task->modes[2] = STARPU_R;
  helper_get_image_shape_multifreq_task->handles[3] = shape_h;
  helper_get_image_shape_multifreq_task->modes[3] = STARPU_W;
  helper_get_image_shape_multifreq_task->synchronous = 1;
  //struct starpu_task *helper_get_image_shape_multifreq_task = create_task(&helper_get_image_shape_multifreq_cl, helper_get_image_shape_multifreq_h);
 
  status_code = starpu_task_submit(helper_get_image_shape_multifreq_task);

	Image *gleam_model;		//Image (GLEAM model)
//	printf("Vis_slices = %d,  npixel = %d, cellsize = %e\n", adv->vis_slices, adv->npixel, adv->cellsize);
	printf("Vis_slices = %d,  npixel = %d, cellsize = %e\n", adv.vis_slices, adv.npixel, adv.cellsize);
	printf("Vis_slices = %d,  npixel = %d, cellsize = %e\n", adv.vis_slices, npixel, cellsize);
  //starpu_task_wait_for_all();
	printf("A shape of the modeled GLEAM image: [ %d, %d, %d, %d]\n", shape[0], shape[1], shape[2], shape[3]);

  printf("Allocating gleam image\n");
	gleam_model = allocate_image(shape);
  printf("confname %s\n",config_name);  
  printf("%d\n",npixel);
  starpu_data_handle_t gleam_model_h;
  printf("confname %s\n",config_name);  
	starpu_variable_data_register(&shape_h, STARPU_MAIN_RAM,
			(uintptr_t)shape, 4*sizeof(int));
	starpu_variable_data_register(&gleam_model_h, STARPU_MAIN_RAM,
			(uintptr_t)gleam_model, sizeof(Image));

//  struct starpu_task *allocate_image_task = starpu_task_create();
//  allocate_image_task->cl = &allocate_image_cl;
//  allocate_image_task->handles[0] = shape_h;
//  allocate_image_task->modes[0] = STARPU_R;
//  allocate_image_task->handles[1] = gleam_model_h;
//  allocate_image_task->modes[1] = STARPU_RW;
//  allocate_image_task->synchronous = 1;
//
//  printf("%d %d %d\n", (&gleam_model->data_shape)[0], (&gleam_model->data_shape)[1], (&gleam_model->data_shape)[2]);
//  printf("confname %c%c%c%c\n",lowconfig->confname[0],lowconfig->confname[1],lowconfig->confname[2],lowconfig->confname[3]);  
//  status_code = starpu_task_submit(allocate_image_task);

  printf("confname %s\n", lowconfig->confname);  
  phasecentre = (vt->phasecentre);
  printf("%d %d %d\n", gleam_model->data_shape[0], gleam_model->data_shape[1], gleam_model->data_shape[2]);
  printf("Creating low test image from gleam model\n");
  starpu_data_handle_t phasecentre_h;
	starpu_variable_data_register(&lowconfig_h, STARPU_MAIN_RAM,
      (uintptr_t)lowconfig, sizeof(ARLConf));
	starpu_variable_data_register(&adv_h, STARPU_MAIN_RAM,
  	(uintptr_t)&(cellsize), sizeof(double));
	starpu_variable_data_register(&npixel_h, STARPU_MAIN_RAM,
			(uintptr_t)&(adv.npixel), sizeof(int));
	starpu_variable_data_register(&phasecentre_h, STARPU_MAIN_RAM,
			(uintptr_t)phasecentre, sizeof(char*));

  struct starpu_task *create_low_test_image_from_gleam_task = starpu_task_create();
  create_low_test_image_from_gleam_task->cl = &create_low_test_image_from_gleam_cl;
  create_low_test_image_from_gleam_task->handles[0] = lowconfig_h;
  create_low_test_image_from_gleam_task->modes[0] = STARPU_R;
  create_low_test_image_from_gleam_task->handles[1] = adv_h;
  create_low_test_image_from_gleam_task->modes[1] = STARPU_R;
  create_low_test_image_from_gleam_task->handles[2] = npixel_h;
  create_low_test_image_from_gleam_task->modes[2] = STARPU_R;
  create_low_test_image_from_gleam_task->handles[3] = phasecentre_h;
  create_low_test_image_from_gleam_task->modes[3] = STARPU_R;
  create_low_test_image_from_gleam_task->handles[4] = gleam_model_h;
  create_low_test_image_from_gleam_task->modes[4] = STARPU_RW;
  create_low_test_image_from_gleam_task->synchronous = 1;

//  printf("%p %p %p %p %d\n", lowconfig_h, adv_h,npixel_h, phasecentre_h, gleam_model->size);
  status_code = starpu_task_submit(create_low_test_image_from_gleam_task);
  starpu_data_unregister(phasecentre_h);
  starpu_data_unregister(npixel_h);
  starpu_data_unregister(adv_h);
//
//	printf("Nrec = %d %d\n", lowconfig->nrec);
//  
//  	arl_create_low_test_image_from_gleam(lowconfig, adv.cellsize, adv.npixel, vt->phasecentre, gleam_model);
//  	arl_create_low_test_image_from_gleam(lowconfig, adv->cellsize, adv->npixel, vt->phasecentre, gleam_model);
//
// FITS file output
	status = mkdir("results", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	status = export_image_to_fits_c(&gleam_model, "!results/ical_c_api-gleam_model.fits");

  printf("Predict blockvisibility\n");
	// predict_function()
//	arl_predict_function_blockvis(lowconfig, vt, gleam_model);
//  starpu_data_handle_t predict_function_blockvis_h[3];
//  SVDR(predict_function_blockvis_h, 0, lowconfig, sizeof(ARLConf));
//  SVDR(predict_function_blockvis_h, 1, &vt, sizeof(ARLVis));
//  SVDR(predict_function_blockvis_h, 2, &gleam_model, sizeof(Image));
  struct starpu_task *predict_function_blockvis_task = starpu_task_create();
  predict_function_blockvis_task->cl = &predict_function_blockvis_cl;
  predict_function_blockvis_task->handles[0] = lowconfig_h;
  predict_function_blockvis_task->modes[0] = STARPU_R;
  predict_function_blockvis_task->handles[1] = vt_h;
  predict_function_blockvis_task->modes[1] = STARPU_RW;
  predict_function_blockvis_task->handles[2] = gleam_model_h;
  predict_function_blockvis_task->modes[2] = STARPU_RW;
  predict_function_blockvis_task->synchronous = 1;
  status_code = starpu_task_submit(predict_function_blockvis_task);
  starpu_data_unregister(gleam_model_h);

//////	// create_gaintable_from_blockvisibility()
//	arl_create_gaintable_from_blockvisibility(lowconfig, vt, gt);
  starpu_data_handle_t gt_h;
	starpu_variable_data_register(&gt_h, STARPU_MAIN_RAM,
			(uintptr_t)gt, sizeof(ARLGt));
////  starpu_data_handle_t create_gaintable_from_blockvisibility_h[3];
////  SVDR(create_gaintable_from_blockvisibility_h, 0, lowconfig, sizeof(ARLConf));
////  SVDR(create_gaintable_from_blockvisibility_h, 1, vt , sizeof(ARLVis));
////  SVDR(create_gaintable_from_blockvisibility_h, 2, &gt, sizeof(ARLGt));
  
  printf("Create gaintable from blockvisibility\n");
  struct starpu_task *create_gaintable_from_blockvisibility_task = starpu_task_create();
  create_gaintable_from_blockvisibility_task->cl = &create_gaintable_from_blockvisibility_cl;
  create_gaintable_from_blockvisibility_task->handles[0] = lowconfig_h;
  create_gaintable_from_blockvisibility_task->modes[0] = STARPU_R;
  create_gaintable_from_blockvisibility_task->handles[1] = vt_h;
  create_gaintable_from_blockvisibility_task->modes[1] = STARPU_R;
  create_gaintable_from_blockvisibility_task->handles[2] = gt_h;
  create_gaintable_from_blockvisibility_task->modes[2] = STARPU_W;
  create_gaintable_from_blockvisibility_task->synchronous = 1;
  status_code = starpu_task_submit(create_gaintable_from_blockvisibility_task);
//  starpu_data_unregister(lowconfig_h);
//  starpu_data_unregister(vt_h);
//  starpu_data_unregister(gt_h);

////	// simulate_gaintable()
//	arl_simulate_gaintable(lowconfig, gt);
////  starpu_data_handle_t simulate_gaintable_h[2];
////  SVDR(simulate_gaintable_h, 0, lowconfig, sizeof(ARLConf));
////  SVDR(simulate_gaintable_h, 1, &gt, sizeof(ARLGt));
//  
  printf("rmax%e\n", lowconfig->rmax);  

  struct starpu_task *simulate_gaintable_task = starpu_task_create();
  simulate_gaintable_task->cl = &simulate_gaintable_cl;
  simulate_gaintable_task->handles[0] = lowconfig_h;
  simulate_gaintable_task->modes[0] = STARPU_R;
  simulate_gaintable_task->handles[1] = gt_h;
  simulate_gaintable_task->modes[1] = STARPU_RW;
  status_code = starpu_task_submit(simulate_gaintable_task);
////
////	// apply_gaintable()
//	arl_apply_gaintable(lowconfig, vt, gt, vt_gt, 1);
////  starpu_data_handle_t apply_gaintable_h[5];
////  SVDR(apply_gaintable_h, 0, lowconfig, sizeof(ARLConf));
////  SVDR(apply_gaintable_h, 1, vt, sizeof(ARLVis));
////  SVDR(apply_gaintable_h, 2, gt, sizeof(ARLGt));
////  SVDR(apply_gaintable_h, 3, vt_gt, sizeof(ARLVis));
////  SVDR(apply_gaintable_h, 4, &inv_in, sizeof(int));
//  starpu_data_handle_t vt_gt_h;
//  starpu_data_handle_t inv_h;
//	starpu_variable_data_register(&vt_gt_h, STARPU_MAIN_RAM,
//			(uintptr_t)vt_gt, sizeof(ARLVis));
//	starpu_variable_data_register(&inv_h, STARPU_MAIN_RAM,
//			(uintptr_t)&inv_in, sizeof(int));
//  
//  
//  struct starpu_task *apply_gaintable_task = starpu_task_create();
//  apply_gaintable_task->cl = &apply_gaintable_cl;
//  apply_gaintable_task->handles[0] = lowconfig_h;
//  apply_gaintable_task->modes[0] = STARPU_R;
//  apply_gaintable_task->handles[1] = vt_h;
//  apply_gaintable_task->modes[1] = STARPU_R;
//  apply_gaintable_task->handles[2] = gt_h;
//  apply_gaintable_task->modes[2] = STARPU_RW;
//  apply_gaintable_task->handles[3] = vt_gt_h;
//  apply_gaintable_task->modes[3] = STARPU_RW;
//  apply_gaintable_task->handles[4] = inv_h;
//  apply_gaintable_task->modes[4] = STARPU_R;
//  //_task->modes[] = STARPU_;
//  status_code = starpu_task_submit(apply_gaintable_task);
//
//	// Create a "model" image with nchan = 1
//	for(i = 0; i< 4; i++) {
//		shape1[i] = shape[i];
//		}
//	shape1[0] = 1;
//	model = allocate_image(shape1);
//
//	// create_image_from_blockvisibility()
////	arl_create_image_from_blockvisibility(lowconfig, vt, adv->cellsize, adv->npixel, vt->phasecentre, model);
////  starpu_data_handle_t create_image_from_blockvisibility_h[6];
////  SVDR(create_image_from_blockvisibility_h, 0, lowconfig, sizeof(ARLConf));
////  SVDR(create_image_from_blockvisibility_h, 1, vt, sizeof(ARLVis));
////  SVDR(create_image_from_blockvisibility_h, 2, &adv->cellsize, sizeof(double));
////  SVDR(create_image_from_blockvisibility_h, 3, &adv->npixel, sizeof(int));
////  SVDR(create_image_from_blockvisibility_h, 4, &vt->phasecentre, sizeof(char*));
////  SVDR(create_image_from_blockvisibility_h, 5, &model, sizeof(Image));
//  starpu_data_handle_t model_h;  
//	starpu_variable_data_register(&model_h, STARPU_MAIN_RAM,
//			(uintptr_t)&model, sizeof(Image));
//
//  struct starpu_task *create_image_from_blockvisibility_task = starpu_task_create();
//  create_image_from_blockvisibility_task->cl = &create_image_from_blockvisibility_cl;
//  create_image_from_blockvisibility_task->handles[0] = lowconfig_h;
//  create_image_from_blockvisibility_task->modes[0] = STARPU_R;
//  create_image_from_blockvisibility_task->handles[1] = vt_h;
//  create_image_from_blockvisibility_task->modes[1] = STARPU_R;
//  create_image_from_blockvisibility_task->handles[2] = adv_h;
//  create_image_from_blockvisibility_task->modes[2] = STARPU_R;
//  create_image_from_blockvisibility_task->handles[3] = npixel_h;
//  create_image_from_blockvisibility_task->modes[3] = STARPU_R;
//  create_image_from_blockvisibility_task->handles[4] = phasecentre_h;
//  create_image_from_blockvisibility_task->modes[4] = STARPU_RW;
//  create_image_from_blockvisibility_task->handles[5] = model_h;
//  create_image_from_blockvisibility_task->modes[5] = STARPU_RW;
//
//  status_code = starpu_task_submit(create_image_from_blockvisibility_task);
////
////	// create a "dirty" image with nchan = 1
//	dirty = allocate_image(shape1);
////
////	// invert_function()
////	arl_invert_function_blockvis(lowconfig, vt_gt, model, adv->vis_slices, dirty);
////  SVDR(invert_function_blockvis_h, 0, lowconfig, sizeof(ARLConf));
////  SVDR(invert_function_blockvis_h, 1, vt_gt, sizeof(ARLVis));
////  SVDR(invert_function_blockvis_h, 2, &model, sizeof(Image));
////  SVDR(invert_function_blockvis_h, 3, &adv->vis_slices, sizeof(int));
////  SVDR(invert_function_blockvis_h, 4, &dirty, sizeof(Image));
//  starpu_data_handle_t vis_slices_h;
//  starpu_data_handle_t dirty_h;
//	starpu_variable_data_register(&vis_slices_h, STARPU_MAIN_RAM,
//			(uintptr_t)&adv.vis_slices, sizeof(int));
//	starpu_variable_data_register(&dirty_h, STARPU_MAIN_RAM,
//			(uintptr_t)&dirty, sizeof(Image));
//  
//  struct starpu_task *invert_function_blockvis_task = starpu_task_create();
//  invert_function_blockvis_task->cl = &invert_function_blockvis_cl;
//  invert_function_blockvis_task->handles[0] = lowconfig_h;
//  invert_function_blockvis_task->modes[0] = STARPU_R;
//  invert_function_blockvis_task->handles[1] = vt_gt_h;
//  invert_function_blockvis_task->modes[1] = STARPU_R;
//  invert_function_blockvis_task->handles[2] = model_h;
//  invert_function_blockvis_task->modes[2] = STARPU_R;
//  invert_function_blockvis_task->handles[3] = vis_slices_h;
//  invert_function_blockvis_task->modes[3] = STARPU_R;
//  invert_function_blockvis_task->handles[4] = dirty_h;
//  invert_function_blockvis_task->modes[4] = STARPU_RW;
//
//  status_code = starpu_task_submit(invert_function_blockvis_task);
//
//
//	// ical() - serial version
//	// create images with nchan = 1
//	deconvolved = allocate_image(shape1);
//	residual    = allocate_image(shape1);
//	restored    = allocate_image(shape1);
//
//	// FITS file output
//	status = export_image_to_fits_c(dirty, "!results/ical_c_api-dirty.fits");
//
//	if(unrolled){
//	// The same values as hard-coded in arlwrap.py calls
//		nmajor = 5; 
//		thresh = 0.1;
//		first_selfcal = 10;
//		printf("ical: Performing %d major cycles\n", nmajor);
//	// Allocate temp objects
//		vis_ical  = allocate_vis_data(lowconfig->npol, nvis);							     //ICAL Visibility object (ical.vis)
//		vpred_ical  = allocate_vis_data(lowconfig->npol, nvis);							     //ICAL Visibility object (ical.vispred)
//		vres_ical  = allocate_vis_data(lowconfig->npol, nvis);							     //ICAL Visibility object (ical.visres)
//		bvtmp_ical = allocate_blockvis_data(lowconfig->nant, lowconfig->nfreqs, lowconfig->npol, lowconfig->ntimes); //Blockvisibility vis_ical.blockvis (ical.vis.blockvis)
//		bvtmp2_ical = allocate_blockvis_data(lowconfig->nant, lowconfig->nfreqs, lowconfig->npol, lowconfig->ntimes); //Blockvisibility vpred_ical.blockvis (ical.vispred.blockvis)
//		bvpred_ical = allocate_blockvis_data(lowconfig->nant, lowconfig->nfreqs, lowconfig->npol, lowconfig->ntimes); //Blockvisibility (ical.block_vispred)
//		psf_ical    = allocate_image(shape1);									// Image PSF
//		dirty_ical    = allocate_image(shape1);									// Image dirty (CLEAN loop)
//		cc_ical    = allocate_image(shape1);									// Image restored tmp (CLEAN loop)
//		res_ical    = allocate_image(shape1);									// Image residual tmp (CLEAN loop)
//		if (!(cindex_ical = malloc(cindex_nbytes))) {								     // Cindex vtmp_ical.cindex (ical.vis.cindex)
//			free(cindex_ical);
//			return 1;
//		}
//		if (!(cindex2_ical = malloc(cindex_nbytes))) {								     // Cindex vtmp_ical.cindex (ical.vis.cindex)
//			free(cindex2_ical);
//			return 1;
//		}
//
//		gt_ical = allocate_gt_data(lowconfig->nant, lowconfig->nfreqs, lowconfig->nrec, lowconfig->ntimes);	// GainTable (CLEAN loop)
//
//	// convert_blockvisibility_to_visibility()
////		arl_convert_blockvisibility_to_visibility(lowconfig, vt_gt, vis_ical, cindex_ical, bvtmp_ical);
////  starpu_data_handle_t convert_blockvisibility_to_visibility_h[5];
////  SVDR(convert_blockvisibility_to_visibility_h, 0, lowconfig, sizeof(ARLConf));
////  SVDR(convert_blockvisibility_to_visibility_h, 1, vt_gt, sizeof(ARLVis));
////  SVDR(convert_blockvisibility_to_visibility_h, 2, vis_ical, sizeof(ARLVis));
////  SVDR(convert_blockvisibility_to_visibility_h, 3, cindex_ical, sizeof(long long int));
////  SVDR(convert_blockvisibility_to_visibility_h, 4, bvtmp_ical, sizeof(ARLVis));
//  starpu_data_handle_t vis_ical_h;
//  starpu_data_handle_t cindex_ical_h;
//  starpu_data_handle_t bvtmp_ical_h;
//	starpu_variable_data_register(&vis_ical_h, STARPU_MAIN_RAM,
//			(uintptr_t)vis_ical, sizeof(ARLVis));
//	starpu_variable_data_register(&cindex_ical_h, STARPU_MAIN_RAM,
//			(uintptr_t)&cindex_ical, sizeof(long long int));
//	starpu_variable_data_register(&bvtmp_ical_h, STARPU_MAIN_RAM,
//			(uintptr_t)bvtmp_ical, sizeof(ARLVis));
//  
//  struct starpu_task *convert_blockvisibility_to_visibility_task = starpu_task_create();
//  convert_blockvisibility_to_visibility_task->cl = &convert_blockvisibility_to_visibility_cl;
//  convert_blockvisibility_to_visibility_task->handles[0] = lowconfig_h;
//  convert_blockvisibility_to_visibility_task->modes[0] = STARPU_R;
//  convert_blockvisibility_to_visibility_task->handles[1] = vt_gt_h;
//  convert_blockvisibility_to_visibility_task->modes[1] = STARPU_R;
//  convert_blockvisibility_to_visibility_task->handles[2] = vis_ical_h;
//  convert_blockvisibility_to_visibility_task->modes[2] = STARPU_RW;
//  convert_blockvisibility_to_visibility_task->handles[3] = cindex_ical_h;
//  convert_blockvisibility_to_visibility_task->modes[3] = STARPU_RW;
//  convert_blockvisibility_to_visibility_task->handles[4] = bvtmp_ical_h;
//  convert_blockvisibility_to_visibility_task->modes[4] = STARPU_RW;
//
//  status_code = starpu_task_submit(convert_blockvisibility_to_visibility_task);
////
////	// copy_visibility (blockvis)
////		arl_copy_blockvisibility(lowconfig, vt_gt, bvpred_ical, 0);
////  starpu_data_handle_t copy_blockvisibility_h[4];
////  SVDR(copy_blockvisibility_h, 0, lowconfig, sizeof(ARLConf));
////  SVDR(copy_blockvisibility_h, 1, vt_gt, sizeof(ARLVis));
////  SVDR(copy_blockvisibility_h, 2, bvpred_ical, sizeof(ARLVis));
////  SVDR(copy_blockvisibility_h, 3, &zero_in, sizeof(int));
//  starpu_data_handle_t bvpred_ical_h;
//  starpu_data_handle_t zero_in_h;
//	starpu_variable_data_register(&bvpred_ical_h, STARPU_MAIN_RAM,
//			(uintptr_t)bvpred_ical, sizeof(ARLVis));
//	starpu_variable_data_register(&zero_in_h, STARPU_MAIN_RAM,
//			(uintptr_t)&zero_in, sizeof(int));
//  
//  struct starpu_task *copy_blockvisibility_task = starpu_task_create();
//  copy_blockvisibility_task->cl = &copy_blockvisibility_cl;
//  copy_blockvisibility_task->handles[0] = lowconfig_h;
//  copy_blockvisibility_task->modes[0] = STARPU_R;
//  copy_blockvisibility_task->handles[1] = vt_gt_h;
//  copy_blockvisibility_task->modes[1] = STARPU_R;
//  copy_blockvisibility_task->handles[2] = bvpred_ical_h;
//  copy_blockvisibility_task->modes[2] = STARPU_RW;
//  copy_blockvisibility_task->handles[3] = zero_in_h;
//  copy_blockvisibility_task->modes[3] = STARPU_RW;
//  status_code = starpu_task_submit(copy_blockvisibility_task);
////
////	// convert_blockvisibility_to_visibility()
////	// Re-using cindex_ical, bvtmp_ical
////		arl_convert_blockvisibility_to_visibility(lowconfig, bvpred_ical, vpred_ical, cindex2_ical, bvtmp2_ical);
// // starpu_data_handle_t [];
////  SVDR(convert_blockvisibility_to_visibility_h, 0, lowconfig, sizeof(ARLConf));
////  SVDR(convert_blockvisibility_to_visibility_h, 1, bvpred_ical, sizeof(ARLVis));
////  SVDR(convert_blockvisibility_to_visibility_h, 2, vpred_ical, sizeof(ARLVis));
////  SVDR(convert_blockvisibility_to_visibility_h, 3, cindex2_ical, sizeof(long long int));
////  SVDR(convert_blockvisibility_to_visibility_h, 4, bvtmp2_ical, sizeof(ARLVis));
//  starpu_data_handle_t vpred_ical_h;
//  starpu_data_handle_t cindex2_ical_h;
//  starpu_data_handle_t bvtmp2_ical_h;
//	starpu_variable_data_register(&vpred_ical_h, STARPU_MAIN_RAM,
//			(uintptr_t)vpred_ical, sizeof(ARLVis));
//	starpu_variable_data_register(&cindex2_ical_h, STARPU_MAIN_RAM,
//			(uintptr_t)&cindex2_ical, sizeof(long long int));
//	starpu_variable_data_register(&bvtmp2_ical_h, STARPU_MAIN_RAM,
//			(uintptr_t)bvtmp2_ical, sizeof(ARLVis));
//  
//  struct starpu_task *convert_blockvisibility_to_visibility_task2 = starpu_task_create();
//  convert_blockvisibility_to_visibility_task->cl = &convert_blockvisibility_to_visibility_cl;
//  convert_blockvisibility_to_visibility_task->handles[0] = lowconfig_h;
//  convert_blockvisibility_to_visibility_task->modes[0] = STARPU_RW;
//  convert_blockvisibility_to_visibility_task->handles[1] = bvpred_ical_h;
//  convert_blockvisibility_to_visibility_task->modes[1] = STARPU_R;
//  convert_blockvisibility_to_visibility_task->handles[2] = vpred_ical_h;
//  convert_blockvisibility_to_visibility_task->modes[2] = STARPU_RW;
//  convert_blockvisibility_to_visibility_task->handles[3] = cindex2_ical_h;
//  convert_blockvisibility_to_visibility_task->modes[3] = STARPU_RW;
//  convert_blockvisibility_to_visibility_task->handles[4] = bvtmp2_ical_h;
//  convert_blockvisibility_to_visibility_task->modes[4] = STARPU_RW;
//
//  status_code = starpu_task_submit(convert_blockvisibility_to_visibility_task2);
////
////	// Set vpred_ical.data to zero
////		arl_set_visibility_data_to_zero(lowconfig, vpred_ical);
//  
//  struct starpu_task *set_visibility_data_to_zero_task = starpu_task_create();
//  set_visibility_data_to_zero_task->cl = &set_visibility_data_to_zero_cl;
//  set_visibility_data_to_zero_task->handles[0] = lowconfig_h;
//  set_visibility_data_to_zero_task->modes[0] = STARPU_R;
//  set_visibility_data_to_zero_task->handles[1] = vpred_ical_h;
//  set_visibility_data_to_zero_task->modes[1] = STARPU_RW;
//  //starpu_task_declare_deps_array(set_visibility_data_to_zero_task, 1, &convert_blockvisibility_to_visibility_task2);
//
//  status_code = starpu_task_submit(set_visibility_data_to_zero_task);
//
////	// copy_visibility (vis)
////		arl_copy_visibility(lowconfig, vpred_ical, vres_ical, 1);
//  zero_in = 1;
////  SVDR(copy_blockvisibility_h, 0, lowconfig, sizeof(ARLConf));
////  SVDR(copy_blockvisibility_h, 1, vpred_ical, sizeof(ARLVis));
////  SVDR(copy_blockvisibility_h, 2, vres_ical, sizeof(ARLVis));
////  SVDR(copy_blockvisibility_h, 3, &zero_in, sizeof(int));
//  starpu_data_handle_t vres_ical_h;
//	starpu_variable_data_register(&vres_ical_h, STARPU_MAIN_RAM,
//			(uintptr_t)vres_ical, sizeof(ARLVis));
//  
//  struct starpu_task *copy_blockvisibility_task2 = starpu_task_create();
//  copy_blockvisibility_task->cl = &copy_blockvisibility_cl;
//  copy_blockvisibility_task->handles[0] = lowconfig_h;
//  copy_blockvisibility_task->modes[0] = STARPU_R;
//  copy_blockvisibility_task->handles[1] = vpred_ical_h;
//  copy_blockvisibility_task->modes[1] = STARPU_R;
//  copy_blockvisibility_task->handles[2] = vres_ical_h;
//  copy_blockvisibility_task->modes[2] = STARPU_R;
//  copy_blockvisibility_task->handles[3] = zero_in_h;
//  copy_blockvisibility_task->modes[3] = STARPU_RW;
//
//  status_code = starpu_task_submit(copy_blockvisibility_task2);
////
////	// predict_function()
////		arl_predict_function_ical(lowconfig, vpred_ical, model, bvtmp2_ical, cindex2_ical, adv->vis_slices);
////  starpu_data_handle_t predict_function_ical_h[6];
////  SVDR(predict_function_ical_h, 0, lowconfig, sizeof(ARLConf));
////  SVDR(predict_function_ical_h, 1, vpred_ical, sizeof(ARLVis));
////  SVDR(predict_function_ical_h, 2, &model, sizeof(Image));
////  SVDR(predict_function_ical_h, 3, bvtmp2_ical, sizeof(ARLVis));
////  SVDR(predict_function_ical_h, 4, &cindex2_ical, sizeof(long long int));
////  SVDR(predict_function_ical_h, 5, &adv->vis_slices, sizeof(int));
//  
//  struct starpu_task *predict_function_ical_task = starpu_task_create();
//  predict_function_ical_task->cl = &predict_function_ical_cl;
//  predict_function_ical_task->handles[0] = lowconfig_h;
//  predict_function_ical_task->modes[0] = STARPU_R;
//  predict_function_ical_task->handles[1] = vpred_ical_h;
//  predict_function_ical_task->modes[1] = STARPU_RW;
//  predict_function_ical_task->handles[2] = model_h;
//  predict_function_ical_task->modes[2] = STARPU_R;
//  predict_function_ical_task->handles[3] = bvtmp2_ical_h;
//  predict_function_ical_task->modes[3] = STARPU_RW;
//  predict_function_ical_task->handles[4] = cindex2_ical_h;
//  predict_function_ical_task->modes[4] = STARPU_RW;
//  predict_function_ical_task->handles[5] = vis_slices_h;
//  predict_function_ical_task->modes[5] = STARPU_RW;
//
//  status_code = starpu_task_submit(predict_function_ical_task);
////
////	// convert_visibility_to_blockvisibility()
////		arl_convert_visibility_to_blockvisibility(lowconfig, vpred_ical, bvtmp2_ical, cindex2_ical, bvpred_ical);
////  starpu_data_handle_t convert_visibility_to_blockvisibility_h[5];
////  SVDR(convert_visibility_to_blockvisibility_h, 0, lowconfig, sizeof(ARLConf));
////  SVDR(convert_visibility_to_blockvisibility_h, 1, vpred_ical, sizeof(ARLVis));
////  SVDR(convert_visibility_to_blockvisibility_h, 2, bvtmp2_ical, sizeof(ARLVis));
////  SVDR(convert_visibility_to_blockvisibility_h, 3, cindex2_ical, sizeof(long long int));
////  SVDR(convert_visibility_to_blockvisibility_h, 4, bvpred_ical, sizeof(ARLVis));
//  
//  struct starpu_task *convert_visibility_to_blockvisibility_task = starpu_task_create();
//  convert_visibility_to_blockvisibility_task->cl = &convert_visibility_to_blockvisibility_cl;
//  convert_visibility_to_blockvisibility_task->handles[0] = lowconfig_h;
//  convert_visibility_to_blockvisibility_task->modes[0] = STARPU_R;
//  convert_visibility_to_blockvisibility_task->handles[1] = vpred_ical_h;
//  convert_visibility_to_blockvisibility_task->modes[1] = STARPU_R;
//  convert_visibility_to_blockvisibility_task->handles[2] = bvtmp2_ical_h;
//  convert_visibility_to_blockvisibility_task->modes[2] = STARPU_R;
//  convert_visibility_to_blockvisibility_task->handles[3] = cindex2_ical_h;
//  convert_visibility_to_blockvisibility_task->modes[3] = STARPU_R;
//  convert_visibility_to_blockvisibility_task->handles[4] = bvpred_ical_h;
//  convert_visibility_to_blockvisibility_task->modes[4] = STARPU_RW;
//
//  status_code = starpu_task_submit(convert_visibility_to_blockvisibility_task);
////
////	// Subtract visibility data to find residuals
////        // vres = vtmp - vpred : 0 = add, 1 = subtract, 2 = mult, 3 = divide, else sets to zero
////		arl_manipulate_visibility_data(lowconfig, vis_ical, vpred_ical, vres_ical, 1); 
////  starpu_data_handle_t manipulate_visibility_data_h[5];
////  SVDR(manipulate_visibility_data_h, 0, lowconfig, sizeof(ARLConf));
////  SVDR(manipulate_visibility_data_h, 1, vis_ical, sizeof(ARLVis));
////  SVDR(manipulate_visibility_data_h, 2, &vpred_ical, sizeof(ARLVis));
////  SVDR(manipulate_visibility_data_h, 3, &vres_ical, sizeof(ARLVis));
////  SVDR(manipulate_visibility_data_h, 4, &op, sizeof(int));
//  starpu_data_handle_t op_h;
//	starpu_variable_data_register(&op_h, STARPU_MAIN_RAM,
//			(uintptr_t)&op, sizeof(int));
//  
//  struct starpu_task *manipulate_visibility_data_task = starpu_task_create();
//  manipulate_visibility_data_task->cl = &manipulate_visibility_data_cl;
//  manipulate_visibility_data_task->handles[0] = lowconfig_h;
//  manipulate_visibility_data_task->modes[0] = STARPU_R;
//  manipulate_visibility_data_task->handles[1] = vis_ical_h;
//  manipulate_visibility_data_task->modes[1] = STARPU_R;
//  manipulate_visibility_data_task->handles[2] = vpred_ical_h;
//  manipulate_visibility_data_task->modes[2] = STARPU_R;
//  manipulate_visibility_data_task->handles[3] = vres_ical_h;
//  manipulate_visibility_data_task->modes[3] = STARPU_RW;
//  manipulate_visibility_data_task->handles[4] = op_h;
//  manipulate_visibility_data_task->modes[3] = STARPU_RW;
//
//  status_code = starpu_task_submit(manipulate_visibility_data_task);
////	// arl_invert_function_ical() (extra parameters in **kwargs -TBS later)
////		arl_invert_function_ical(lowconfig, vres_ical, model, adv->vis_slices, dirty_ical);
////  starpu_data_handle_t invert_function_ical_h[5];
////  SVDR(invert_function_ical_h, 0, lowconfig, sizeof(ARLConf));
////  SVDR(invert_function_ical_h, 1, vres_ical, sizeof(ARLVis));
////  SVDR(invert_function_ical_h, 2, &model, sizeof(Image));
////  SVDR(invert_function_ical_h, 3, &adv->vis_slices, sizeof(int));
////  SVDR(invert_function_ical_h, 4, &dirty_ical, sizeof(Image));
//  starpu_data_handle_t dirty_ical_h;
//	starpu_variable_data_register(&dirty_ical_h, STARPU_MAIN_RAM,
//			(uintptr_t)&dirty_ical, sizeof(Image));
//  
//  struct starpu_task *invert_function_ical_task = starpu_task_create();
//  invert_function_ical_task->cl =  &invert_function_ical_cl;
//  invert_function_ical_task->handles[0] = lowconfig_h;
//  invert_function_ical_task->modes[0] =  STARPU_R;
//  invert_function_ical_task->handles[1] = vres_ical_h;
//  invert_function_ical_task->modes[1] =  STARPU_R;
//  invert_function_ical_task->handles[2] = model_h;
//  invert_function_ical_task->modes[2] =  STARPU_R;
//  invert_function_ical_task->handles[3] = vis_slices_h;
//  invert_function_ical_task->modes[3] =  STARPU_R;
//  invert_function_ical_task->handles[4] = dirty_ical_h;
//  invert_function_ical_task->modes[4] =  STARPU_RW;
//
//  status_code = starpu_task_submit(invert_function_ical_task);
//
////	// arl_invert_function_psf() (extra parameters in **kwargs -TBS later)
////		arl_invert_function_psf(lowconfig, vres_ical, model, adv->vis_slices, psf_ical);
////  starpu_data_handle_t invert_function_psf_h[5];
////  SVDR(invert_function_psf_h, 0, &lowconfig, sizeof(ARLConf));
////  SVDR(invert_function_psf_h, 1, vres_ical, sizeof(ARLVis));
////  SVDR(invert_function_psf_h, 2, &model, sizeof(Image));
////  SVDR(invert_function_psf_h, 3, &adv->vis_slices, sizeof(int));
////  SVDR(invert_function_psf_h, 4, &psf_ical, sizeof(Image));
//  starpu_data_handle_t psf_ical_h;
//	starpu_variable_data_register(&psf_ical_h, STARPU_MAIN_RAM,
//			(uintptr_t)&psf_ical, sizeof(Image));
//  
//  struct starpu_task *invert_function_psf_task = starpu_task_create();
//  invert_function_psf_task->cl = &invert_function_psf_cl;
//  invert_function_psf_task->handles[0] = lowconfig_h;
//  invert_function_psf_task->modes[0] = STARPU_R;
//  invert_function_psf_task->handles[1] = vres_ical_h;
//  invert_function_psf_task->modes[1] = STARPU_R;
//  invert_function_psf_task->handles[2] = model_h;
//  invert_function_psf_task->modes[2] = STARPU_R;
//  invert_function_psf_task->handles[3] = vis_slices_h;
//  invert_function_psf_task->modes[3] = STARPU_RW;
//  invert_function_psf_task->handles[4] = psf_ical_h;
//  invert_function_psf_task->modes[4] = STARPU_RW;
//
//  status_code = starpu_task_submit(invert_function_psf_task);
////	// CLEAN major cycles		
////		for(i = 0; i< nmajor; i++){
////			printf("ical: Start of major cycle %d of %d\n", i, nmajor);
////		// arl_deconvolve_cube_ical() with hard-coded **kwargs	
////  starpu_data_handle_t deconvolve_cube_ical_h[4];
////  SVDR(deconvolve_cube_ical_h, 0, &dirty_ical, sizeof(Image));
////  SVDR(deconvolve_cube_ical_h, 1, &psf_ical, sizeof(Image));
////  SVDR(deconvolve_cube_ical_h, 2, &cc_ical, sizeof(Image));
////  SVDR(deconvolve_cube_ical_h, 3, &res_ical, sizeof(Image));
//  starpu_data_handle_t cc_ical_h;
//	starpu_variable_data_register(&cc_ical_h, STARPU_MAIN_RAM,
//			(uintptr_t)&cc_ical, sizeof(Image));
//  starpu_data_handle_t res_ical_h;
//	starpu_variable_data_register(&res_ical_h, STARPU_MAIN_RAM,
//			(uintptr_t)&res_ical, sizeof(Image));
//  
//  
//  struct starpu_task *deconvolve_cube_ical_task = starpu_task_create();
//  deconvolve_cube_ical_task->cl = &deconvolve_cube_ical_cl;
//  deconvolve_cube_ical_task->handles[0] = dirty_ical_h;
//  deconvolve_cube_ical_task->modes[0] = STARPU_R;
//  deconvolve_cube_ical_task->handles[1] = psf_ical_h;
//  deconvolve_cube_ical_task->modes[1] = STARPU_R;
//  deconvolve_cube_ical_task->handles[2] = cc_ical_h;
//  deconvolve_cube_ical_task->modes[2] = STARPU_R;
//  deconvolve_cube_ical_task->handles[3] = res_ical_h;
//  deconvolve_cube_ical_task->modes[3] = STARPU_RW;
//  status_code = starpu_task_submit(deconvolve_cube_ical_task);
////			arl_deconvolve_cube_ical(dirty_ical, psf_ical, cc_ical, res_ical);
////		// add cc_ical into the model
////			arl_add_to_model(model, cc_ical);
////  starpu_data_handle_t add_to_model_h[2];
////  SVDR(add_to_model_h, 0, &model, sizeof(Image));
////  SVDR(add_to_model_h, 1, &cc_ical, sizeof(Image));
//  
//  struct starpu_task *add_to_model_task = starpu_task_create();
//  add_to_model_task->cl = &add_to_model_cl;
//  add_to_model_task->handles[0] = model_h;
//  add_to_model_task->modes[0] = STARPU_R;
//  add_to_model_task->handles[1] = cc_ical_h;
//  add_to_model_task->modes[1] = STARPU_RW;
//
//  status_code = starpu_task_submit(add_to_model_task);
////		// Set vpred_ical.data to zero
////			arl_set_visibility_data_to_zero(lowconfig, vpred_ical);
////  starpu_data_handle_t set_visibility_data_to_zero_h[2];
////  SVDR(set_visibility_data_to_zero_h, 0, lowconfig, sizeof(ARLConf));
////  SVDR(set_visibility_data_to_zero_h, 1, vpred_ical, sizeof(ARLVis));
//  
//  struct starpu_task *set_visibility_data_to_zero_task2 = starpu_task_create();
//  set_visibility_data_to_zero_task2->cl = &set_visibility_data_to_zero_cl;
//  set_visibility_data_to_zero_task2->handles[0] = lowconfig_h;
//  set_visibility_data_to_zero_task2->modes[0] = STARPU_R;
//  set_visibility_data_to_zero_task2->handles[1] = vpred_ical_h;
//  set_visibility_data_to_zero_task2->modes[1] = STARPU_RW;
//
//  status_code = starpu_task_submit(set_visibility_data_to_zero_task2);
////		// predict_function()
////			arl_predict_function_ical(lowconfig, vpred_ical, model, bvtmp2_ical, cindex2_ical, adv->vis_slices);
////  starpu_data_handle_t predict_function_ical_h[6];
////  SVDR(predict_function_ical_h, 0, lowconfig, sizeof(ARLConf));
////  SVDR(predict_function_ical_h, 1, vpred_ical, sizeof(ARLVis));
////  SVDR(predict_function_ical_h, 2, &model, sizeof(Image));
////  SVDR(predict_function_ical_h, 3, bvtmp2_ical, sizeof(ARLVis));
////  SVDR(predict_function_ical_h, 4, &cindex2_ical, sizeof(long long int));
////  SVDR(predict_function_ical_h, 5, &adv->vis_slices, sizeof(int));
//   
//  struct starpu_task *predict_function_ical_task2 = starpu_task_create();
//  predict_function_ical_task2->cl = &predict_function_ical_cl;
//  predict_function_ical_task2->handles[0] = lowconfig_h;
//  predict_function_ical_task2->modes[0] = STARPU_R;
//  predict_function_ical_task2->handles[1] = vpred_ical_h;
//  predict_function_ical_task2->modes[1] = STARPU_R;
//  predict_function_ical_task2->handles[2] = model_h;
//  predict_function_ical_task2->modes[2] = STARPU_R;
//  predict_function_ical_task2->handles[3] = bvtmp2_ical_h;
//  predict_function_ical_task2->modes[3] = STARPU_R;
//  predict_function_ical_task2->handles[4] = cindex2_ical_h;
//  predict_function_ical_task2->modes[4] = STARPU_R;
//  predict_function_ical_task2->handles[5] = vis_slices_h;
//  predict_function_ical_task2->modes[5] = STARPU_RW;
//
//  status_code = starpu_task_submit(predict_function_ical_task2);
////		// if doselfcal (currently not working, to be replaced with calibrate_function)
/////*			if(i >= first_selfcal) {
////				printf("ical: Performing selfcalibration\n");
////				// convert_visibility_to_blockvisibility()
////				arl_convert_visibility_to_blockvisibility(lowconfig, vpred_ical, bvtmp2_ical, cindex2_ical, bvpred_ical);
////				// arl_solve_gaintable()
////				arl_solve_gaintable_ical(lowconfig, vt_gt, bvpred_ical, gt_ical, adv->vis_slices);
////				// arl_apply_gaintable_ical() and re-write vt_gt (ical::block_vis)
////				arl_apply_gaintable_ical(lowconfig, vt_gt, gt_ical, 0);
////				// convert_blockvisibility_to_visibility()
////				arl_convert_blockvisibility_to_visibility(lowconfig, vt_gt, vis_ical, cindex_ical, bvtmp_ical);
////			}
////*/			
////        	// vres = vtmp - vpred : 0 = add, 1 = subtract, 2 = mult, 3 = divide, else sets to zero
////			arl_manipulate_visibility_data(lowconfig, vis_ical, vpred_ical, vres_ical, 1); 
////		// arl_invert_function_ical() (extra parameters in **kwargs -TBS later)
////			arl_invert_function_ical(lowconfig, vres_ical, model, adv->vis_slices, dirty_ical);
////
////		// ToDo - loop break on threshold
////
////			printf("ical: End of major cycle %d\n", i);
////
////		}
////		printf("ical: End of major cycles\n");
////		arl_restore_cube_ical(model, psf_ical, dirty_ical, restored);
////
////    starpu_task_wait_for_all();
////
////	// FITS file output, unrolled version of the files
//		status = export_image_to_fits_c(model, 		"!results/ical_c_api_u_deconvolved.fits"); 
//		status = export_image_to_fits_c(dirty_ical,	"!results/ical_c_api_u_residual.fits");
//		status = export_image_to_fits_c(restored, 	"!results/ical_c_api_u_restored.fits");
////	// Cleaning up temp objects	
//		bvtmp_ical 		= destroy_vis(bvtmp_ical);
//		bvtmp2_ical 		= destroy_vis(bvtmp2_ical);
//		bvpred_ical		= destroy_vis(bvpred_ical);
//		vis_ical 		= destroy_vis(vis_ical);
//		vpred_ical 		= destroy_vis(vpred_ical);
//		vres_ical 		= destroy_vis(vres_ical);
//		psf_ical		= destroy_image(psf_ical);
//		dirty_ical		= destroy_image(dirty_ical);
//		cc_ical			= destroy_image(cc_ical);
//		res_ical		= destroy_image(res_ical);
//		gt_ical 		= destroy_gt(gt_ical);
//
//		free(cindex_ical);
//		free(cindex2_ical);
//
//	} else {	
//
////		model = destroy_image(model);
////		model = allocate_image(shape1);
////		arl_create_image_from_blockvisibility(lowconfig, vt_gt, adv->cellsize, adv->npixel, vt_gt->phasecentre, model);
//
////		arl_ical(lowconfig, vt_gt, model, adv->vis_slices, deconvolved, residual, restored);
//	// FITS file output, normal versions of the files
//    starpu_task_wait_for_all();
//		status = export_image_to_fits_c(deconvolved, 	"!results/ical_c_api_deconvolved.fits");
//		status = export_image_to_fits_c(residual, 	"!results/ical_c_api_residual.fits");
//		status = export_image_to_fits_c(restored, 	"!results/ical_c_api_restored.fits");
//	}
//
  starpu_task_wait_for_all();

//  starpu_data_unregister(lowconfig_h);
//  starpu_data_unregister(vt_h);
//  starpu_data_unregister(adv_h);
//  starpu_data_unregister(npixel_h);
//  starpu_data_unregister(shape_h);
  starpu_data_unregister(gleam_model_h);
//  starpu_data_unregister(gt_h);
//  starpu_data_unregister(phasecentre_h);

	//gt 		= destroy_gt(&gt);

  starpu_shutdown();

	// Cleaning up
//	gleam_model 	= destroy_image(gleam_model);
//	model		= destroy_image(model);
//	dirty		= destroy_image(dirty);
//	deconvolved	= destroy_image(deconvolved);
//	residual	= destroy_image(residual);
//	restored	= destroy_image(restored);
//	vt 		= destroy_vis(vt);
////	vtpredicted 	= destroy_vis(vtpredicted);
//	vt_predictfunction = destroy_vis(vt_predictfunction);
//	vt_gt 		= destroy_vis(vt_gt);
//	free(cindex_predict);
//	free(shape);
//	free(shape1);
////  free(adv);
//  free(lowconfig);



	return 0;

}
