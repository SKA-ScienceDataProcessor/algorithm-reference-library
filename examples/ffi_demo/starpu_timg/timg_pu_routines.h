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

#ifndef TIMG_PU_ROUTINES_H
#define TIMG_PU_ROUTINES_H
#include <starpu.h>

#include "../src/arlwrap.h"

// I'm not typing this out every time
#define SVGP(x) STARPU_VARIABLE_GET_PTR(buffers[x])
#define SVDR(handle, i, var, size) starpu_variable_data_register(&handle[i], STARPU_MAIN_RAM, (uintptr_t)var, size);


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

struct starpu_task *create_task(struct starpu_codelet *kernel, starpu_data_handle_t *handles);
void create_task_and_submit(struct starpu_codelet *kernel, starpu_data_handle_t *handles);
#endif
