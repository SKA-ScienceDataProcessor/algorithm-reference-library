#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <math.h>

#include <starpu.h>

struct starpu_arlvis_if
{
  void *cpu_ptr;
  size_t nvis;
  int npol;
  void *data;
  char *phasecentre; 
};

void arlvis_register_data_handle(starpu_data_handle_t handle, unsigned home_node, void *data_interface);
starpu_ssize_t arlvis_allocate_data_on_node(void *data_interface, unsigned node);
size_t arlvis_get_size();
void arlvis_footprint();
void * arlvis_handle_to_pointer(starpu_data_handle_t handle, unsigned node);
void * starpu_arlvis_get_cpu_ptr(starpu_data_handle_t handle);
size_t starpu_arlvis_get_nvis(starpu_data_handle_t handle);
int starpu_arlvis_get_npol(starpu_data_handle_t handle);
void *starpu_arlvis_get_data(starpu_data_handle_t handle);
char *starpu_arlvis_get_phase(starpu_data_handle_t handle);

#define STARPU_ARLVIS_GET_CPU_PTR(interface)  \
  (((struct starpu_arlvis_if *)(interface))->cpu_ptr)
#define STARPU_ARLVIS_GET_NVIS(interface)  \
  (((struct starpu_arlvis_if *)(interface))->nvis)
#define STARPU_ARLVIS_GET_NPOL(interface)  \
  (((struct starpu_arlvis_if *)(interface))->npol)
#define STARPU_ARLVIS_GET_DATA(interface)  \
  (((struct starpu_arlvis_if *)(interface))->data)
#define STARPU_ARLVIS_GET_PHASECENTRE(interface)  \
  (((struct starpu_arlvis_if *)(interface))->phasecentre)

