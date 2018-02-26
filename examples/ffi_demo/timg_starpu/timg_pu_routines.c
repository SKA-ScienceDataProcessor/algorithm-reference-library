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

#include "timg_pu_routines.h"

// Simple interfaces that 
void pu_create_test_image(void **buffers, void *cl_arg)
{
	arl_create_test_image(STARPU_VARIABLE_GET_PTR(buffers[0]), *((double*)STARPU_VARIABLE_GET_PTR(buffers[1])),
			STARPU_VARIABLE_GET_PTR(buffers[2]), STARPU_VARIABLE_GET_PTR(buffers[3]));
}

void pu_create_visibility(void **buffers, void *cl_arg)
{
	arl_create_visibility(SVGP(0), SVGP(1));
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
