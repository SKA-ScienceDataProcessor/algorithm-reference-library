
The scripts in this directory construct SLURM scripts for execution on CSD3. These scrips have grown organically and 
could definitely be improved upon.

1. make_csd3_tests.sh makes a set of SLURM scripts, iterating over some parameters such as rmax, nnodes, etc.
1. make_csd3_tests.sh calls make_pipelines_timings.py to construct a SLURM script working from the 
template submit_csd3_template.

   make_pipelines-timings.sh HOSTNAME NUMBER_NODES NUMBER_PROCS_PER_NODE NUMBER_FREQUENCY_WINDOWS CONTEXT NUMBER_THREADS MEMORY SERIAL RMAX 
 
  * HOSTNAME csd3
  * NUMBER_NODES number of nodes to be used
  * NUMBER_PROCS_PER_NODE number of workers on each node
  * NUMBER_FREQUENCY_WINDOWS number of frequency windows to create
  * CONTEXT imaging context timeslice or wstack
  * NUMBER_THREADS number of threads per worker, usually 1
  * MEMORY memory per node in GB
  * SERIAL predict and invert are serial versions True or False
  * RMAX Maximum radius of telescope in metres

1. submit_csd3_template is the template for the SLURM script. The variables above (HOSTNAME, NUMBER_NODES, etc.) are 
used in the template

The name of the SLURM script encodes some of the 
parameters. For example:

submit_csd3_2nodes_16procspernode_32ntasks_64nfreqwin_1threads_384000memory_timeslicecontext_Trueserial_1200rmax

