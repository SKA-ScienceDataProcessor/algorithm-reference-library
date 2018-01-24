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

#include <starpu.h>

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
