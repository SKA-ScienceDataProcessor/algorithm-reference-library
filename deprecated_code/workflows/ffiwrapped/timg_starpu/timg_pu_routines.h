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

#ifndef __TIMG_PU_ROUTINES_H__
#define __TIMG_PU_ROUTINES_H__
#include <fitsio.h>
#include <starpu.h>

#include "../include/arlwrap.h"

// I'm not typing this out every time
#define SVGP(x) STARPU_VARIABLE_GET_PTR(buffers[x])
#define SVDR(handle, i, var, size) starpu_variable_data_register(&handle[i], STARPU_MAIN_RAM, (uintptr_t)var, size);

void pu_create_test_image(void **buffers, void *cl_arg);

void pu_create_visibility(void **buffers, void *cl_arg);

void pu_predict_2d(void **buffers, void *cl_arg);

void pu_create_from_visibility(void **buffers, void *cl_args);

void pu_invert_2d(void **buffers, void *cl_args);

void pu_deconvolve_cube(void **buffers, void *cl_args);

void pu_restore_cube(void **buffers, void *cl_args);


/* Example kernel codelet: calls create_visibility, specifies number of buffers
 * (=arguments), and set argument read/write modes
 */
extern struct starpu_codelet create_visibility_cl;

extern struct starpu_codelet create_test_image_cl;

extern struct starpu_codelet predict_2d_cl;

extern struct starpu_codelet create_from_visibility_cl;

extern struct starpu_codelet invert_2d_cl;

extern struct starpu_codelet deconvolve_cube_cl;

extern struct starpu_codelet restore_cube_cl;

struct starpu_task *create_task(struct starpu_codelet *kernel, starpu_data_handle_t *handles);
void create_task_and_submit(struct starpu_codelet *kernel, starpu_data_handle_t *handles);

#endif
