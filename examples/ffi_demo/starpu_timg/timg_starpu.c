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
 */

#include <Python.h>
#include <starpu.h>
#include <stdarg.h>
#include <stdio.h>

#include "../src/arlwrap.h"

#include <fitsio.h>


// I'm not typing this out every time
#define SVGP(x) STARPU_VARIABLE_GET_PTR(buffers[x])
#define SVDR(handle, i, var, size) starpu_variable_data_register(&handle[i], STARPU_MAIN_RAM, (uintptr_t)var, size);

/* Export image to FITS */
/* Assuming nx*ny*nfreq */
/* ToDo - add polarization and wcs */
int export_image_to_fits_c(Image *im, char * filename) {
	int status = 0, exists;
	fitsfile *fptr;       /* pointer to the FITS file; defined in fitsio.h */
	long  fpixel = 1, naxis = 4, nelements;
	long naxes[4];

	naxes[0] = im->data_shape[3];
	naxes[1] = im->data_shape[2];
	naxes[2] = im->data_shape[1];
	naxes[3] = im->data_shape[0];

	fits_file_exists(filename, &exists, &status); /* check if the file exists */

	if(exists != 0) {
		fits_open_file(&fptr, filename, READWRITE, &status); /* open existed file */
	}
	else {
		fits_create_file(&fptr, filename, &status);   /* create new file */
	}

	/* Create the primary array image  */
	fits_create_img(fptr, DOUBLE_IMG, naxis, naxes, &status);
	nelements = naxes[0] * naxes[1] * naxes[2] * naxes[3];          /* number of pixels to write */
	/* Write the array of integers to the image */
	fits_write_img(fptr, TDOUBLE, fpixel, nelements, im->data, &status);
	fits_close_file(fptr, &status);            /* close the file */
	fits_report_error(stderr, status);  /* print out any error messages */
	return status;
}

Image *allocate_image(int *shape)
{
	int i;
	Image *image = malloc(sizeof(Image));

	image->size = 1;//shape[0]*shape[1]*shape[2]*shape[3];

	for(i=0; i<4; i++) {
		image->data_shape[i] = shape[i];
		image->size *= shape[i];
	}

	image->data = calloc(image->size,sizeof(double));
	image->wcs = calloc(2997,sizeof(char));
	image->polarisation_frame = calloc(115,sizeof(char));

	return image;
}

void pu_create_visibility(void **buffers, void *cl_arg)
{
	arl_create_visibility(SVGP(0), SVGP(1));
}

void pu_create_test_image(void **buffers, void *cl_arg)
{
	arl_create_test_image(STARPU_VARIABLE_GET_PTR(buffers[0]), *((double*)STARPU_VARIABLE_GET_PTR(buffers[1])),
			STARPU_VARIABLE_GET_PTR(buffers[2]), STARPU_VARIABLE_GET_PTR(buffers[3]));
}

void pu_predict_2d(void **buffers, void *cl_arg)
{
	arl_predict_2d(SVGP(0), SVGP(1), SVGP(2));
}

void pu_create_from_visibility(void **buffers, void *cl_args)
{
	arl_create_image_from_visibility(SVGP(0), SVGP(1));
}

void pu_invert_2d(void **buffers, void *cl_args)
{
	arl_invert_2d(SVGP(0), SVGP(1), *((bool*)SVGP(2)), SVGP(3), SVGP(4));
}

void pu_deconvolve_cube(void **buffers, void *cl_args)
{
	arl_deconvolve_cube(SVGP(0), SVGP(1), SVGP(2), SVGP(3));
}

void pu_restore_cube(void **buffers, void *cl_args)
{
	arl_restore_cube(SVGP(0), SVGP(1), SVGP(2), SVGP(3));
}


/* Example kernel codelet: calls create_visibility, specifies number of buffers
 * (=arguments), and set argument read/write modes
 */
struct starpu_codelet create_visibility_cl = {
	.cpu_funcs = { pu_create_visibility },
	.name = "pu_create_visibility",
	.nbuffers = 2,
	.modes = { STARPU_R, STARPU_W }
};

struct starpu_codelet create_test_image_cl = {
	.cpu_funcs = { pu_create_test_image },
	.name = "pu_create_test_image",
	.nbuffers = 4,
	.modes = { STARPU_R, STARPU_R, STARPU_W, STARPU_W }
};

struct starpu_codelet predict_2d_cl = {
	.cpu_funcs = { pu_predict_2d },
	.name = "pu_predict_2d",
	.nbuffers = 3,
	.modes = { STARPU_R, STARPU_R, STARPU_W }
};

struct starpu_codelet create_from_visibility_cl = {
	.cpu_funcs = { pu_create_from_visibility },
	.name = "pu_create_from_visibility",
	.nbuffers = 2,
	.modes = { STARPU_R, STARPU_W }
};

struct starpu_codelet invert_2d_cl = {
	.cpu_funcs = { pu_invert_2d },
	.name = "pu_invert_2d",
	.nbuffers = 5,
	.modes = { STARPU_R, STARPU_R, STARPU_R, STARPU_W, STARPU_RW }
};

struct starpu_codelet deconvolve_cube_cl = {
	.cpu_funcs = { pu_deconvolve_cube },
	.name = "pu_deconvolve_cube",
	.nbuffers = 4,
	.modes = { STARPU_R, STARPU_R, STARPU_RW, STARPU_RW }
};

struct starpu_codelet restore_cube_cl = {
	.cpu_funcs = { pu_restore_cube },
	.name = "pu_restore_cube",
	.nbuffers = 4,
	.modes = { STARPU_R, STARPU_R, STARPU_RW, STARPU_RW }
};

/* Simple task submission. Assumes one buffer per handle, nbuffers and modes
 * specified in kernel codelet
 */

struct starpu_task *create_task(struct starpu_codelet *kernel, starpu_data_handle_t *handles)
{
	int i;
	struct starpu_task *task = starpu_task_create();
	task->cl = kernel;

	for(i = 0; i < kernel->nbuffers; i++) {
		task->handles[i] = handles[i];
	}

	return task;

}

void create_task_and_submit(struct starpu_codelet *kernel, starpu_data_handle_t *handles)
{
	struct starpu_task *task = create_task(kernel, handles);
	starpu_task_submit(task);
}

int main(int argc, char *argv[]) {

	starpu_init(NULL);

	Py_Initialize();

	// We need this macro (+ END macro), otherwise starpu's pthreads will deadlock
	// when trying to get the Python interpreter lock
	Py_BEGIN_ALLOW_THREADS

	/* BEGIN setup_stolen_from_ffi_demo */
	int *shape = malloc(4*sizeof(int));
	int status;
	int nvis=1;

	double *times = calloc(1,sizeof(double));
	double *freq = malloc(1*sizeof(double));
	double *channel_bandwidth = malloc(1*sizeof(double));
	freq[0] = 1e8;
	channel_bandwidth[0] = 1e6;
	double cellsize = 0.0005;
	char config_name[] = "LOWBD2-CORE";

	ARLVis *vt = malloc(sizeof(ARLVis));
	ARLVis *vtmodel = malloc(sizeof(ARLVis));
	ARLVis *vtmp = malloc(sizeof(ARLVis));

	ARLConf *lowconfig = malloc(sizeof(ARLConf));

	ant_t nb;

	// Find out the number of the antennas and the baselines, keep in nb structure
	nb.nbases = 1;
	helper_get_nbases(config_name, &nb);
	// Assigning configuraion values
	lowconfig->confname = config_name;
	lowconfig->pc_ra = 15.0;
	lowconfig->pc_dec = -45.0;
	lowconfig->times = times;
	lowconfig->ntimes = 1;
	lowconfig->freqs = freq;
	lowconfig->nfreqs = 1;
	lowconfig->channel_bandwidth = channel_bandwidth;
	lowconfig->nchanwidth = 1;
	lowconfig->nbases = nb.nbases;
	lowconfig->npol = 1;
	// Find out the number of visibilities
	nvis = (lowconfig->nbases)*(lowconfig->nfreqs)*(lowconfig->ntimes);
	printf("nvis = %d\n", nvis);

	vt->nvis = nvis;
	vt->npol = lowconfig->npol;

	// malloc to ARLDataVisSize
	vt->data = malloc((72+32*vt->npol)*vt->nvis * sizeof(char));
	vtmp->data = malloc((72+32*vt->npol)*vt->nvis * sizeof(char));
	vtmodel->data = malloc((72+32*vt->npol)*vt->nvis * sizeof(char));
	printf("%d, %d, %p\n", vt->npol, vt->nvis, vt);

	/* malloc data for phasecentre pickle.
	 * TODO un-hardcode size
	 */
	vt->phasecentre = malloc(5000*sizeof(char));
	vtmp->phasecentre = malloc(5000*sizeof(char));
	vtmodel->phasecentre = malloc(5000*sizeof(char));

	// TODO check all mallocs
	if (!vt->data || !vtmp->data || !vtmodel->data ||
			!vt->phasecentre || !vtmp->phasecentre || !vtmodel->phasecentre) {
		fprintf(stderr, "Malloc error\n");
		exit(1);
	}
	/* END setup_stolen_from_ffi_demo */

	starpu_data_handle_t create_visibility_h[2];
	starpu_variable_data_register(&create_visibility_h[0], STARPU_MAIN_RAM,
			(uintptr_t)lowconfig, sizeof(ARLConf));
	starpu_variable_data_register(&create_visibility_h[1], STARPU_MAIN_RAM,
			(uintptr_t)vt, sizeof(ARLVis));

	struct starpu_task *vis_task = create_task(&create_visibility_cl, create_visibility_h);
	starpu_task_submit(vis_task);

	helper_get_image_shape(freq, cellsize, shape);

	Image *model = allocate_image(shape);
	Image *m31image = allocate_image(shape);
	Image *dirty = allocate_image(shape);
	Image *psf = allocate_image(shape);
	Image *comp = allocate_image(shape);
	Image *residual = allocate_image(shape);
	Image *restored = allocate_image(shape);

	starpu_data_handle_t test_image_h[4];
	starpu_variable_data_register(&test_image_h[0], STARPU_MAIN_RAM,
			(uintptr_t)freq, sizeof(double*));
	starpu_variable_data_register(&test_image_h[1], STARPU_MAIN_RAM,
			(uintptr_t)&cellsize, sizeof(double));
	starpu_variable_data_register(&test_image_h[2], STARPU_MAIN_RAM,
			(uintptr_t)(vt->phasecentre), sizeof(char*));
	starpu_variable_data_register(&test_image_h[3], STARPU_MAIN_RAM,
			(uintptr_t)m31image, sizeof(Image));
	struct starpu_task *test_img_task = create_task(&create_test_image_cl, test_image_h);


	// For some reason (TODO: find out why) StarPU is not getting data
	// dependencies right, so we need to explicitly tell it about task
	// dependencies instead.
	starpu_task_declare_deps_array(test_img_task, 1, &vis_task);

	starpu_task_submit(test_img_task);

	starpu_data_handle_t pred_handle[3];
	SVDR(pred_handle, 0, vt, sizeof(ARLVis));
	SVDR(pred_handle, 1, m31image, sizeof(Image));
	SVDR(pred_handle, 2, vtmp, sizeof(ARLVis));

	struct starpu_task *pred_task = create_task(&predict_2d_cl, pred_handle);
	starpu_task_declare_deps_array(pred_task, 1, &test_img_task);
	starpu_task_submit(pred_task);

	starpu_data_handle_t create_from_vis_handle[2];
	SVDR(create_from_vis_handle, 0, vtmp, sizeof(ARLVis));
	SVDR(create_from_vis_handle, 1, model, sizeof(Image));
	struct starpu_task *create_from_vis_task = create_task(&create_from_visibility_cl, create_from_vis_handle);
	starpu_task_declare_deps_array(create_from_vis_task, 1, &pred_task);
	starpu_task_submit(create_from_vis_task);

	bool invert_false = false;
	bool invert_true = true;

	double *sumwt = malloc(sizeof(double));

	starpu_data_handle_t invert_2d_dirty_handle[5];
	SVDR(invert_2d_dirty_handle, 0, vt, sizeof(ARLVis));
	SVDR(invert_2d_dirty_handle, 1, model, sizeof(Image));
	SVDR(invert_2d_dirty_handle, 2, &invert_false, sizeof(bool));
	SVDR(invert_2d_dirty_handle, 3, dirty, sizeof(Image));
	SVDR(invert_2d_dirty_handle, 4, sumwt, sizeof(double));

	struct starpu_task *invert_2d_dirty_task = create_task(&invert_2d_cl, invert_2d_dirty_handle);
	starpu_task_declare_deps_array(invert_2d_dirty_task, 1, &create_from_vis_task);
	starpu_task_submit(invert_2d_dirty_task);

	starpu_data_handle_t invert_2d_psf_handle[5];
	SVDR(invert_2d_psf_handle, 0, vt, sizeof(ARLVis));
	SVDR(invert_2d_psf_handle, 1, model, sizeof(Image));
	SVDR(invert_2d_psf_handle, 2, &invert_true, sizeof(bool));
	SVDR(invert_2d_psf_handle, 3, psf, sizeof(Image));
	SVDR(invert_2d_psf_handle, 4, sumwt, sizeof(double));

	struct starpu_task *invert_2d_psf_task = create_task(&invert_2d_cl, invert_2d_psf_handle);
	starpu_task_declare_deps_array(invert_2d_psf_task, 1, &invert_2d_dirty_task);
	starpu_task_submit(invert_2d_psf_task);

	starpu_data_handle_t deconvolve_cube_handle[4];
	SVDR(deconvolve_cube_handle, 0, dirty, sizeof(Image));
	SVDR(deconvolve_cube_handle, 1, psf, sizeof(Image));
	SVDR(deconvolve_cube_handle, 2, comp, sizeof(Image));
	SVDR(deconvolve_cube_handle, 3, residual, sizeof(Image));

	struct starpu_task *deconvolve_cube_task = create_task(&deconvolve_cube_cl, deconvolve_cube_handle);
	starpu_task_declare_deps_array(deconvolve_cube_task, 1, &invert_2d_psf_task);
	starpu_task_submit(deconvolve_cube_task);

	starpu_data_handle_t restore_cube_handle[4];
	SVDR(restore_cube_handle, 0, comp, sizeof(Image));
	SVDR(restore_cube_handle, 1, psf, sizeof(Image));
	SVDR(restore_cube_handle, 2, residual, sizeof(Image));
	SVDR(restore_cube_handle, 3, restored, sizeof(Image));

	struct starpu_task *restore_cube_task = create_task(&restore_cube_cl, restore_cube_handle);
	starpu_task_declare_deps_array(restore_cube_task, 1, &deconvolve_cube_task);
	starpu_task_submit(restore_cube_task);


	// === Terminate ===
	starpu_task_wait_for_all();

	// FITS files output
	status = mkdir("results", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	status = export_image_to_fits_c(m31image, "results/m31image.fits");
	status = export_image_to_fits_c(dirty, "results/dirty.fits");
	status = export_image_to_fits_c(psf, "results/psf.fits");
	status = export_image_to_fits_c(residual, "results/residual.fits");
	status = export_image_to_fits_c(restored, "results/restored.fits");
	status = export_image_to_fits_c(comp, "results/solution.fits");

	starpu_shutdown();
	//verify phasecentre was correctly written

	Py_END_ALLOW_THREADS
	return 0;
}
