/* 
 * timg_starpu.c
 *
 * Implements a basic TIMG pipeline using StarPU and the ARL C Wrappers.
 *
 * Very much a work-in-progress with a lot of duplicated code.
 * TODO: Some code sourced directly from ffi_demo.c, need to create separate
 * source/header for all helper routines.
 *
 * Author: Arjen Tamerus <at748@cam.ac.uk>
 * Author: Chris Hadjigeorgiou <ch741@cam.ac.uk>
 */

#include <stdarg.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <starpu.h>

#include "../../../ffiwrappers/include/arlwrap.h"
#include "../../../ffiwrappers/include/wrap_support.h"
#include "timg_pu_routines.h"

int main(int argc, char *argv[]) {
	/* BEGIN setup_stolen_from_ffi_demo */
	int *shape = malloc(4*sizeof(int));
	int status;
	int nvis;

	double cellsize = 5e-4;
	char config_name[] = "LOWBD2-CORE";

	ARLVis *vt;
	ARLVis *vtmp;

	ARLConf *lowconfig;

	starpu_init(NULL);

	// Initialise the Python interpreter and GIL, and other ARL dependencies
	arl_initialize();


	lowconfig = allocate_arlconf_default(config_name);

	nvis = lowconfig->nbases * lowconfig->nfreqs * lowconfig->ntimes;

	vt = allocate_vis_data(lowconfig->npol, nvis);
	vtmp = allocate_vis_data(lowconfig->npol, nvis);

	/* END setup_stolen_from_ffi_demo */

	starpu_data_handle_t lowconfig_h;
	starpu_data_handle_t vt_h;

	starpu_variable_data_register(&lowconfig_h, STARPU_MAIN_RAM,
			(uintptr_t)lowconfig, sizeof(ARLConf));
	starpu_variable_data_register(&vt_h, STARPU_MAIN_RAM,
			(uintptr_t)vt, sizeof(ARLVis));

	struct starpu_task *vis_task = starpu_task_create();
	
	vis_task->cl = &create_visibility_cl;
	vis_task->handles[0] = lowconfig_h;
	vis_task->modes[0] = STARPU_R;
	vis_task->handles[1] = vt_h;
	vis_task->modes[1] = STARPU_W;

	//starpu_task_submit(vis_task);
	arl_create_visibility(lowconfig, vt);
	helper_get_image_shape(lowconfig->freqs, cellsize, shape);

	Image *model = allocate_image(shape);
	Image *m31image = allocate_image(shape);
	Image *dirty = allocate_image(shape);
	Image *psf = allocate_image(shape);
	Image *comp = allocate_image(shape);
	Image *residual = allocate_image(shape);
	Image *restored = allocate_image(shape);

	/* Data handles are used by StarPU to pass (pointers to) data to the codelets
	 * at execution time */

	/* For now we are just passing the raw pointers to required data, to the data
	 * handle. Most routines expect pointers at this point, and it is easier to
	 * handle edge cases in the codelets, keeping this main routine clean. */
	starpu_data_handle_t cellsize_h;
	starpu_data_handle_t m31image_h;

	starpu_variable_data_register(&lowconfig_h, STARPU_MAIN_RAM,
			(uintptr_t)lowconfig->freqs, sizeof(double*));
	starpu_variable_data_register(&vt_h, STARPU_MAIN_RAM,
			(uintptr_t)vt->phasecentre, sizeof(char*));
	starpu_variable_data_register(&cellsize_h, STARPU_MAIN_RAM,
			(uintptr_t)&cellsize, sizeof(double));
	starpu_variable_data_register(&m31image_h, STARPU_MAIN_RAM,
			(uintptr_t)m31image, sizeof(Image));

	struct starpu_task *test_img_task = starpu_task_create();
	
	test_img_task->cl = &create_test_image_cl;
	test_img_task->handles[0] = lowconfig_h;
	test_img_task->modes[0] = STARPU_R;
	test_img_task->handles[1] = cellsize_h;
	test_img_task->modes[1] = STARPU_R;
	test_img_task->handles[2] = vt_h;
	test_img_task->modes[2] = STARPU_W;
	test_img_task->handles[3] = m31image_h;
	test_img_task->modes[3] = STARPU_W;
	
	starpu_task_submit(test_img_task);
	
	
	
	starpu_data_handle_t vtmp_h;
	starpu_variable_data_register(&vt_h, STARPU_MAIN_RAM,
			(uintptr_t)vt, sizeof(ARLVis));
	starpu_variable_data_register(&vtmp_h, STARPU_MAIN_RAM,
			(uintptr_t)vtmp, sizeof(Image));

	struct starpu_task *pred_task = starpu_task_create();
	
	pred_task->cl = &predict_2d_proto_cl;
	pred_task->handles[0] = vt_h;
	pred_task->modes[0] = STARPU_R;
	pred_task->handles[1] = m31image_h;
	pred_task->modes[1] = STARPU_R;
	pred_task->handles[2] = vtmp_h;
	pred_task->modes[2] = STARPU_W;
	
	starpu_task_submit(pred_task);
	// Use macros for data registration frome here, to improve readability
	
	starpu_data_handle_t model_h;
	starpu_variable_data_register(&model_h, STARPU_MAIN_RAM,
			(uintptr_t)model, sizeof(Image));

	struct starpu_task *create_from_vis_task = starpu_task_create();
  
	create_from_vis_task->cl = &create_from_visibility_cl;
	create_from_vis_task->handles[0] = vtmp_h;
	create_from_vis_task->modes[0] = STARPU_R;
	create_from_vis_task->handles[1] = model_h;
	create_from_vis_task->modes[1] = STARPU_W;
	starpu_task_submit(create_from_vis_task);

	bool invert_false = false;
	bool invert_true = true;

	double *sumwt = malloc(sizeof(double));

	starpu_data_handle_t invert_false_h;
	starpu_data_handle_t dirty_h;
	starpu_data_handle_t sumwt_h;
	starpu_variable_data_register(&invert_false_h, STARPU_MAIN_RAM,
			(uintptr_t)&invert_false, sizeof(bool));
	starpu_variable_data_register(&dirty_h, STARPU_MAIN_RAM,
			(uintptr_t)dirty, sizeof(Image));
	starpu_variable_data_register(&sumwt_h, STARPU_MAIN_RAM,
			(uintptr_t)sumwt, sizeof(double));
	struct starpu_task *invert_2d_dirty_task = starpu_task_create();
	invert_2d_dirty_task->cl = &invert_2d_cl;
	invert_2d_dirty_task->handles[0] = vt_h ;
	invert_2d_dirty_task->modes[0] = STARPU_R ;
	invert_2d_dirty_task->handles[1] = model_h;
	invert_2d_dirty_task->modes[1] = STARPU_R ;
	invert_2d_dirty_task->handles[2] = invert_false_h;
	invert_2d_dirty_task->modes[2] = STARPU_R ;
	invert_2d_dirty_task->handles[3] = dirty_h;
	invert_2d_dirty_task->modes[3] = STARPU_W ;
	invert_2d_dirty_task->handles[4] = sumwt_h;
	invert_2d_dirty_task->modes[4] = STARPU_RW ;

	starpu_task_submit(invert_2d_dirty_task);
//  if(arl_handle_error() != 0)
//   return -1;

	starpu_data_handle_t invert_true_h;
	starpu_data_handle_t psf_h;
	starpu_variable_data_register(&invert_true_h, STARPU_MAIN_RAM,
			(uintptr_t)&invert_true, sizeof(bool));
	starpu_variable_data_register(&psf_h, STARPU_MAIN_RAM,
			(uintptr_t)psf, sizeof(Image));
  
	struct starpu_task *invert_2d_psf_task = starpu_task_create();

	invert_2d_psf_task->cl = &invert_2d_cl;
	invert_2d_psf_task->handles[0] = vt_h;
	invert_2d_psf_task->modes[0] = STARPU_R ;
	invert_2d_psf_task->handles[1] = model_h;
	invert_2d_psf_task->modes[1] = STARPU_R ;
	invert_2d_psf_task->handles[2] = invert_true_h;
	invert_2d_psf_task->modes[2] = STARPU_R ;
	invert_2d_psf_task->handles[3] = psf_h;
	invert_2d_psf_task->modes[3] = STARPU_W ;
	invert_2d_psf_task->handles[4] = sumwt_h;
	invert_2d_psf_task->modes[4] = STARPU_RW ;

	starpu_task_submit(invert_2d_psf_task);

	starpu_data_handle_t comp_h;
	starpu_data_handle_t residual_h;
	starpu_variable_data_register(&comp_h, STARPU_MAIN_RAM,
			(uintptr_t)comp, sizeof(Image));
	starpu_variable_data_register(&residual_h, STARPU_MAIN_RAM,
			(uintptr_t)residual, sizeof(Image));

	struct starpu_task *deconvolve_cube_task = starpu_task_create();
	deconvolve_cube_task->cl = &deconvolve_cube_cl;
	deconvolve_cube_task->handles[0] = dirty_h;
	deconvolve_cube_task->modes[0] = STARPU_R ;
	deconvolve_cube_task->handles[1] = psf_h;
	deconvolve_cube_task->modes[1] = STARPU_R ;
	deconvolve_cube_task->handles[2] = comp_h;
	deconvolve_cube_task->modes[2] = STARPU_RW ;
	deconvolve_cube_task->handles[3] = residual_h;
	deconvolve_cube_task->modes[3] = STARPU_RW ;
  
	starpu_task_submit(deconvolve_cube_task);

	// Set N_ITER > 1 for multithreading test
	// memory doesn't get cleaned, though, so don't set it too high or the 
	// OoM monster will get you!
	starpu_task_wait_for_all();
	#define N_ITER 1
	int i;
	Image *restored_[N_ITER];
	Image *psf_[N_ITER];
	Image *comp_[N_ITER];
	Image *residual_[N_ITER];
	starpu_data_handle_t restore_cube_handle[N_ITER][4];
	starpu_data_handle_t restored_n_h[N_ITER];
	starpu_data_handle_t psf_n_h[N_ITER];
	starpu_data_handle_t comp_n_h[N_ITER];
	starpu_data_handle_t residual_n_h[N_ITER];
	struct starpu_task *restore_cube_task[N_ITER];
	for(i=0; i< N_ITER; i++) {
		comp_[i] = allocate_image(shape);
		memcpy(comp_[i]->data, comp->data, comp->size*sizeof(double));
		memcpy(comp_[i]->wcs, comp->wcs, 2996);
		memcpy(comp_[i]->polarisation_frame, comp->polarisation_frame, 117);
		psf_[i] = allocate_image(shape);
		memcpy(psf_[i]->data, psf->data, psf->size*sizeof(double));
		memcpy(psf_[i]->wcs, psf->wcs, 2996);
		memcpy(psf_[i]->polarisation_frame, psf->polarisation_frame, 117);
		residual_[i] = allocate_image(shape);
		memcpy(residual_[i]->data, residual->data, residual->size*sizeof(double));
		memcpy(residual_[i]->wcs, residual->wcs, 2996);
		memcpy(residual_[i]->polarisation_frame, residual->polarisation_frame, 117);
		restored_[i] = allocate_image(shape);

		starpu_variable_data_register(&comp_n_h[i], STARPU_MAIN_RAM,
		      	(uintptr_t)comp_[i], sizeof(Image));
		starpu_variable_data_register(&psf_n_h[i], STARPU_MAIN_RAM,
		      	(uintptr_t)psf_[i], sizeof(Image));
		starpu_variable_data_register(&residual_n_h[i], STARPU_MAIN_RAM,
		      	(uintptr_t)residual_[i], sizeof(Image));
		starpu_variable_data_register(&restored_n_h[i], STARPU_MAIN_RAM,
			(uintptr_t)restored_[i], sizeof(Image));

		restore_cube_task[i] = starpu_task_create();
		restore_cube_task[i]->cl = &restore_cube_cl;
		restore_cube_task[i]->handles[0] = comp_n_h[i];
		restore_cube_task[i]->modes[0] = STARPU_R ;
		restore_cube_task[i]->handles[1] = psf_n_h[i];
		restore_cube_task[i]->modes[1] = STARPU_R ;
		restore_cube_task[i]->handles[2] = residual_n_h[i];
		restore_cube_task[i]->modes[2] = STARPU_RW ;
		restore_cube_task[i]->handles[3] = restored_n_h[i];
		restore_cube_task[i]->modes[3] = STARPU_RW ;

		starpu_task_submit(restore_cube_task[i]);
	}
	return 0;

	// === Terminate ===

	// Wait for all tasks to complete
	starpu_task_wait_for_all();

	// FITS files output
	status = mkdir("results", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	status = export_image_to_fits_c(m31image, "results/m31image.fits");
	status = export_image_to_fits_c(dirty, "results/dirty.fits");
	status = export_image_to_fits_c(psf, "results/psf.fits");
	status = export_image_to_fits_c(residual, "results/residual.fits");
	status = export_image_to_fits_c(restored_[N_ITER-1], "results/restored.fits");
	status = export_image_to_fits_c(comp, "results/solution.fits");

	//init_interpreters();
	starpu_shutdown();

	// Restore thread state and cleanly shutdown Python
	arl_finalize();

	return 0;
}
