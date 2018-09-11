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

#include "ical_pu_routines.h"

// Simple interfaces that 
void pu_create_test_image(void **buffers, void *cl_arg)
{
	arl_create_test_image(STARPU_VARIABLE_GET_PTR(buffers[0]), *((double*)STARPU_VARIABLE_GET_PTR(buffers[1])),
			STARPU_VARIABLE_GET_PTR(buffers[2]), STARPU_VARIABLE_GET_PTR(buffers[3]));
}

void pu_create_blockvisibility(void **buffers, void *cl_arg)
{
	arl_create_blockvisibility(SVGP(0), SVGP(1));
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

void pu_advise_wide_field(void **buffers, void *cl_args)
{
  arl_advise_wide_field(SVGP(0), SVGP(1), SVGP(2));
}

void pu_helper_get_image_shape_multifreq(void **buffers, void *cl_args)
{
  helper_get_image_shape_multifreq(STARPU_VARIABLE_GET_PTR(buffers[0]),*(double*)SVGP(1),*(int*)STARPU_VARIABLE_GET_PTR(buffers[2]), STARPU_VARIABLE_GET_PTR(buffers[3]));
}

void pu_allocate_image(void **buffers, void *cl_args)
{
  int i;
  Image* tmp = (Image *)allocate_image(SVGP(0));
  Image *out = (Image *)((uintptr_t)SVGP(1));
  *out = *tmp;
	for(i=0; i<4; i++) {
		out->data_shape[i] = tmp->data_shape[i];
		out->size = tmp->size;
	}
  printf("%d %d %d \n", out->size, out->data_shape[0], out->data_shape[2]);
  printf("%d %d %d \n", ((Image *)SVGP(1))->size, ((Image *)SVGP(1))->data_shape[0], ((Image *)SVGP(1))->data_shape[2]);
}

void pu_create_low_test_image_from_gleam(void **buffers, void *cl_args)
{
  Image *out_img = ((Image *)SVGP(4));
  printf("size %d %d %d %d\n",out_img->data_shape[0],out_img->data_shape[1], out_img->data_shape[2], out_img->data_shape[3]);
  arl_create_low_test_image_from_gleam((ARLConf *)SVGP(0), *(double *)SVGP(1), *(int *)SVGP(2), (char*)SVGP(3), out_img);
}

void pu_predict_function_blockvis(void **buffers, void *cl_args)
{
  printf("confname %s\n", ((ARLConf *)SVGP(0))->confname);
  Image* img = (Image*)SVGP(2);
  arl_predict_function_blockvis((ARLConf *)SVGP(0), SVGP(1), img);
}

void pu_create_gaintable_from_blockvisibility(void **buffers, void *cl_args)
{
  printf("create gaintable confname %s\n", ((ARLConf *)SVGP(0))->confname);
  arl_create_gaintable_from_blockvisibility(SVGP(0), SVGP(1), (ARLGt *)SVGP(2));
  printf("confname %s\n", ((ARLConf *)SVGP(0))->confname);
}

void pu_simulate_gaintable(void **buffers, void *cl_args)
{
  arl_simulate_gaintable(SVGP(0),(ARLGt *)SVGP(1));
}

void pu_apply_gaintable(void **buffers, void *cl_args)
{
  arl_apply_gaintable(SVGP(0), SVGP(1), SVGP(2), SVGP(3), SVGP(4));
}

void pu_create_image_from_blockvisibility(void **buffers, void *cl_args)
{
  arl_create_image_from_blockvisibility(SVGP(0), SVGP(1), SVGP(2), SVGP(3), SVGP(4), SVGP(5));
}

void pu_invert_function_blockvis(void **buffers, void *cl_args)
{
  arl_invert_function_blockvis(SVGP(0), SVGP(1), SVGP(2), SVGP(3), SVGP(4));
}

void pu_convert_blockvisibility_to_visibility(void **buffers, void *cl_args)
{
  arl_convert_blockvisibility_to_visibility(SVGP(0), SVGP(1), SVGP(2), SVGP(3), SVGP(4));
}

void pu_copy_blockvisibility(void **buffers, void *cl_args)
{
  arl_copy_blockvisibility(SVGP(0), SVGP(1), SVGP(2), SVGP(3));
}

void pu_add_to_model(void **buffers, void *cl_args)
{
  arl_add_to_model(SVGP(0), SVGP(1));
}

void pu_set_visibility_data_to_zero(void **buffers, void *cl_args)
{
  arl_set_visibility_data_to_zero(SVGP(0), SVGP(1));
}

void pu_predict_function_ical(void **buffers, void *cl_args)
{
  arl_predict_function_ical(SVGP(0), SVGP(1), SVGP(2), SVGP(3), SVGP(4), SVGP(5));
}

void pu_convert_visibility_to_blockvisibility(void **buffers, void *cl_args)
{
  arl_convert_visibility_to_blockvisibility(SVGP(0), SVGP(1), SVGP(2), SVGP(3), SVGP(4));
}

void pu_manipulate_visibility_data(void **buffers, void *cl_args)
{
  arl_manipulate_visibility_data(SVGP(0), SVGP(1), SVGP(2), SVGP(3), SVGP(4));
}

void pu_invert_function_ical(void **buffers, void *cl_args)
{
  arl_invert_function_ical(SVGP(0), SVGP(1), SVGP(2), SVGP(3), SVGP(4));
}

void pu_invert_function_psf(void **buffers, void *cl_args)
{
  arl_invert_function_psf(SVGP(0), SVGP(1), SVGP(2), SVGP(3), SVGP(4));
}

void pu_deconvolve_cube_ical(void **buffers, void *cl_args)
{
  arl_deconvolve_cube_ical(SVGP(0), SVGP(1), SVGP(2), SVGP(3));
}

//void pu_set_visibility_data_to_zero(void **buffers, void *cl_args)
//{
// set_visibility_data_to_zero(SVGP(0), SVGP(1));
//}


//void pu_(void **buffers, void *cl_args)
//{
//  (SVGP(0), SVGP(1), SVGP(2), SVGP(3));
//}

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
	int status = starpu_task_submit(task);
}
