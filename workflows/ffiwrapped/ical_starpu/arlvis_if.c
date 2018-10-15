#include "arlvis_if.h"

 void arlvis_register_data_handle(starpu_data_handle_t handle, unsigned home_node, void *data_interface)
{
  struct starpu_arlvis_if *arlvis_if = (struct starpu_arlvis_if *) data_interface;

  unsigned node;
  for(node = 0; node < STARPU_MAXNODES; node++)
  {
    struct starpu_arlvis_if *local_if = 
      (struct starpu_arlvis_if *) starpu_data_get_interface_on_node(handle, node);
//    if(arlvis_if->nvis >= STARPU_MAXNODES && (arlvis_if->nvis % STARPU_MAXNODES) == 0)
//    {
//      local_if->nvis = (arlvis_if->nvis)/STARPU_MAXNODES;
//      local_if->npol = arlvis_if->npol;
//      local_if->data = &arlvis_if->data[local_if->nvis*node];
//      local_if->phasecentre = arlvis_if->phasecentre;
//    }
//    else
//    { 
      local_if->nvis = arlvis_if->nvis;
      local_if->npol = arlvis_if->npol;
      local_if->data = arlvis_if->data;
      local_if->phasecentre = arlvis_if->phasecentre;
//    }
//    if(node == home_node)
      local_if->cpu_ptr = arlvis_if->cpu_ptr;
//    else
//      local_if->cpu_ptr = NULL;
   
  }
}
starpu_ssize_t arlvis_allocate_data_on_node(void *data_interface, unsigned node)
{
  struct starpu_arlvis_if *arlvis_if = (struct starpu_arlvis_if *) data_interface;

  void *addr_data;
  char *addr_phasecentre;
  starpu_ssize_t data_memory = (80+(32*arlvis_if->npol))*arlvis_if->nvis *sizeof(char);
  starpu_ssize_t phase_memory = 5000 * sizeof(char);
  addr_data = starpu_malloc_on_node(node, data_memory);
  if(!addr_data)
    return -ENOMEM;
  addr_phasecentre = starpu_malloc_on_node(node,phase_memory);
  if(!addr_phasecentre)
  {
    starpu_free_on_node(node, (uintptr_t) addr_data, data_memory);
    return -ENOMEM;
  } 

  arlvis_if->data = addr_data;
  arlvis_if->phasecentre = addr_phasecentre;
  return data_memory+phase_memory;
}
struct starpu_data_copy_methods arlvis_copy_methods = 
{
  .ram_to_ram = NULL
};

size_t arlvis_get_size(starpu_data_handle_t handle)
{
  size_t size;
	struct starpu_arlvis_if *arlvis_if;

	arlvis_if = (struct starpu_arlvis_if *)
				starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);
	size = (80+(32*arlvis_if->npol))*arlvis_if->nvis *sizeof(char)+ 5000 * sizeof(char)+sizeof(int)+sizeof(void*)+sizeof(size_t);
  return size;
}

 void arlvis_footprint()
{

}

void * arlvis_handle_to_pointer(starpu_data_handle_t handle, unsigned node)
{
  struct starpu_arlvis_if *arlvis_if = 
    (struct starpu_arlvis_if *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

	switch(starpu_node_get_kind(node))
	{
		case STARPU_CPU_RAM:
      return arlvis_if->cpu_ptr;
		case STARPU_MAIN_RAM:
      return arlvis_if->cpu_ptr;
    default:
      assert(0);
  }
}static struct starpu_data_interface_ops if_arlvis_ops =
{
        .register_data_handle = arlvis_register_data_handle,
        .allocate_data_on_node = arlvis_allocate_data_on_node,
        .copy_methods = &arlvis_copy_methods,
        .get_size = arlvis_get_size,
        .footprint = arlvis_footprint,
        .interfaceid = STARPU_UNKNOWN_INTERFACE_ID,
        .interface_size = sizeof(struct starpu_arlvis_if),
        .handle_to_pointer = arlvis_handle_to_pointer
};

void * starpu_arlvis_get_cpu_ptr(starpu_data_handle_t handle)
{
  struct starpu_arlvis_if *arlvis_if = 
    (struct starpu_arlvis_if *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

  return arlvis_if->cpu_ptr;
}

size_t starpu_arlvis_get_nvis(starpu_data_handle_t handle)
{
  struct starpu_arlvis_if *arlvis_if = 
    (struct starpu_arlvis_if *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

  return arlvis_if->nvis;
}

int starpu_arlvis_get_npol(starpu_data_handle_t handle)
{
  struct starpu_arlvis_if *arlvis_if = 
    (struct starpu_arlvis_if *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

  return arlvis_if->npol;
}

void *starpu_arlvis_get_data(starpu_data_handle_t handle)
{
  struct starpu_arlvis_if *arlvis_if = 
    (struct starpu_arlvis_if *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

  return arlvis_if->data;
}

char *starpu_arlvis_get_phase(starpu_data_handle_t handle)
{
  struct starpu_arlvis_if *arlvis_if = 
    (struct starpu_arlvis_if *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

  return arlvis_if->phasecentre;
}

void starpu_arlvis_data_register(starpu_data_handle_t *handle,
    unsigned home_node, void * cpu_ptr, size_t nvis, int npol, void *data, char* phasecentre)
{
  struct starpu_arlvis_if arlvis =
  {
    .cpu_ptr = cpu_ptr,
    .nvis = nvis,
    .npol = npol,
    .data = data,
    .phasecentre = phasecentre
  };

  if(if_arlvis_ops.interfaceid == STARPU_UNKNOWN_INTERFACE_ID)
  {
    if_arlvis_ops.interfaceid = starpu_data_interface_get_next_id();
  }
  
  starpu_data_register(handle, home_node, &arlvis, &if_arlvis_ops);
}


