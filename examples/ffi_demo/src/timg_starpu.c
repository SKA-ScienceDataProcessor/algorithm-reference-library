#include <Python.h>
#include <starpu.h>
#include <stdarg.h>
#include <stdio.h>

#include "arlwrap.h"

// starpu kernel calling arl wrapper
void pu_create_visibility(void *buffers[], void *cl_arg)
{
	arl_create_visibility((ARLConf *)STARPU_VARIABLE_GET_PTR(buffers[0]), (ARLVis *)STARPU_VARIABLE_GET_PTR(buffers[1]));
}

/* Example kernel codelet: calls create_visibility, specifies number of buffers
 * (=arguments), and set argument read/write modes
 */
struct starpu_codelet create_visibility_cl = {
	.cpu_funcs = { pu_create_visibility },
	.name = "pu_create_visibility",
	.nbuffers = 2,
	.modes = { STARPU_R, STARPU_RW }
};

/* Simple task submission. Assumes one buffer per handle, nbuffers and modes
 * specified in kernel codelet
 */
void create_task_and_submit(struct starpu_codelet *kernel, starpu_data_handle_t *handles, int num_buffers)
{
	int i;
	struct starpu_task *task = starpu_task_create();
	task->cl = kernel;

	for(i = 0; i < kernel->nbuffers; i++) {
		task->handles[i] = handles[i];
	}

	starpu_task_submit(task);

}

int main(int argc, char *argv[]) {

	starpu_init(NULL);

	Py_Initialize();

	// We need this macro (+ END macro), otherwise starpu's pthreads will hang
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

	create_task_and_submit(&create_visibility_cl, create_visibility_h, 2);


	// === Terminate ===
	starpu_task_wait_for_all();
	starpu_shutdown();
	//verify phasecentre was correctly written
	printf("%s\n", vt->phasecentre);

	Py_END_ALLOW_THREADS
	return 0;
}
