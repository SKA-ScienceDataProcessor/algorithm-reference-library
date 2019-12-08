
ARL and DASK
************

ARL uses Dask for distributed processing:

    http://dask.pydata.org/en/latest/

    https://github.com/dask/dask-tutorial

Running ARL and Dask on a single machine is straightforward. First define a graph and then compute it either by
calling the compute method of the graph or by passing the graph to a dask client.

A typical graph will flow from a set of input visibility sets to an image or set of images. In the course
of constructing a graph, we will need to know the data elements and the functions transforming brtween them.
These are well-modeled in ARL.

In order that Dask.delayed processing can be switched on and off, and that the same code is used for Dask and
non-Dask processing, we have wrapped Dask.delayed in :py:mod:`wrappers.arlexecute.execution_support.arlexecute.arlexecute`.
An example is::

        arlexecute.set_client(use_dask=True)
        continuum_imaging_list = \
            continuum_imaging_component(vis_list, model_imagelist=self.model_imagelist, context='2d',
                                        algorithm='mmclean', facets=1,
                                        scales=[0, 3, 10],
                                        niter=1000, fractional_threshold=0.1,
                                        nmoments=2, nchan=self.freqwin,
                                        threshold=2.0, nmajor=5, gain=0.1,
                                        deconvolve_facets=8, deconvolve_overlap=16,
                                        deconvolve_taper='tukey')
        clean, residual, restored = arlexecute.compute(continuum_imaging_list, sync=True)

The function :py:mod:`wrappers.arlexecute.execution_support.arlexecute.arlexecute.set_client` must be called
before defining any components. If use_dask is True then a Dask graph is constructed for subsequent execution. If
use_dask is False then the function is called immediately.

The pipeline workflow
:py:mod:`workflows.arlexecute.pipelines.pipeline_components.continuum_imaging_component` is itself assembled using the
:py:mod:`wrappers.arlexecute.execution_support.arlexecute.arlexecute.execute` function.

The functions for creating graphs are:

    - :py:mod:`workflows.arlexecute.support_workflows.imaging_workflows`: Graphs to perform various types of prediction and inversion of visibility data
    - :py:mod:`workflows.arlexecute.image.generic_workflows`: Graphs to perform generic image operations
    - :py:mod:`workflows.arlexecute.visibility.generic_workflows`: Graphs to perform generic visibility perations
    - :py:mod:`workflows.arlexecute.simulation.simulation_workflows`: Graphs to support simulations
    - :py:mod:`workflows.arlexecute.pipelines.pipeline_workflows`: Graphs to implement the canonical pipelines

In addition there are notebooks that use components in workflows/notebooks.

    - simple-dask: Demonstrates generic components
    - imaging-pipelines: Pipeline to run continuum imaging and ICAL pipelines on small LOW observation
    - modelpartition: Model partition calibration

These notebooks are scaled to run on a 2017-era laptop (4 cores, 16GB) but can be changed to larger scales. Both
explicitly create a client and output the URL (usually http://127.0.0.1:8787) for the Dask diagnostics. Of these the
status page is most useful. If you shrink the browser size enough laterally all of the information appears on one
page. The image below shows a typical screen for one of the pipelines:

.. image:: ./dask_global.png
   :scale: 100 %

Using ARL and Dask
==================

Logging is difficult when using distributed processing. Here's a solution that works. At the beginning of your script
 or notebook, define a function to initialize the logger.::

    import logging

    start_time_str = time.ctime().replace(' ', '_')
    results_dir = './results/%s' % start_time_str
    os.makedirs(results_dir, exist_ok=True)

    def init_logging():
        logging.basicConfig(filename='%s/ASKAP_simulation.%d.log' % (results_dir, os.getpid()),
                            filemode='w',
                            format='%(process)s %(asctime)s.%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%a, %d %b %Y %H:%M:%S',
                            level=logging.INFO)

    log = logging.getLogger()
    init_logging()
    logging.info("ASKAP_simulation")

To ensure that the Dask workers get the same setup, you will need to run init_logging() on each worker using the
Client.run() function::

    c=get_dask_Client()
    c.run(init_logging)

or::

    arlexecute.set_client(use_dask=True)
    arlexecute.run(init_logging)

This will produce one directory per execution, and in that directory one log file per worker and one for the master.
You can tail these, etc. This may not be what you might want since it is worker-centric. All tasks run on a given
worker are logged to the same file.


Using ARL and dask on Darwin
============================

Running on a cluster is quite a bit more complicated, mostly because of the ways that clusters are operated. Darwin
uses SLURM for scheduling. There is python binding of DRMAA that could in principle be used to queue the processing.
However in the end, a simple edited job submission script was sufficient.

After quite a bit of experimentation I decided to avoid a virtual environment because of apparent problems using
those on worker nodes.

* PATH=~/python/bin:$PATH
* cd $ARL; pip install --prefix=~/python -r requirements.txt
* pip install --prefix=~/python paramiko

Ensure that the .bashrc file has the same definition as .bash_profile. If not, ssh will give strange errors! The
PYTHONPATH should look like::

    $ echo $PYTHONPATH
    /home/hpccorn1/Code/algorithm-reference-library:/home/hpccorn1/arlenv/lib/python3.5/site-packages

You can start a scheduler and workers by hand. Set the environment variable ARL_DASK_SCHEDULER appropriately::

    export ARL_DASK_SCHEDULER=192.168.2.10:8786

If you do this, remember to start the workers as well. dask-ssh is useful for this::

    c=get_dask_Client(timeout=30)
    c.scheduler_info()

get_dask_Client will look for a scheduler via the environment variable ARL_DASK_SCHEDULER. It that does not exist, it
 will start a Client using the default Dask approach.

On darwin, each node has 16 cores, and each core has 4GB. Usually this is insufficient for ARL and so some cores must be
 not used so the memory can be used by other cores. To run 7 workers and one scheduler on 4 nodes, the SLURM batch
 file should look something like::

    #!/bin/bash
    #!
    #! Dask job script for Darwin (Sandy Bridge, ConnectX3)
    #! Tim Cornwell
    #!

    #!#############################################################
    #!#### Modify the options in this section as appropriate ######
    #!#############################################################

    #! sbatch directives begin here ###############################
    #! Name of the job:
    #SBATCH -J SDP_ARL
    #! Which project should be charged:
    #SBATCH -A SKA-SDP
    #! How many whole nodes should be allocated?
    #SBATCH --nodes=4
    #! How many (MPI) tasks will there be in total? (<= nodes*16)
    #SBATCH --ntasks=8
    #! How much wallclock time will be required?
    #SBATCH --time=00:10:00
    #! What types of email messages do you wish to receive?
    #SBATCH --mail-type=FAIL
    #! Uncomment this to prevent the job from being requeued (e.g. if
    #! interrupted by node failure or system downtime):
    ##SBATCH --no-requeue

    #! Do not change:
    #SBATCH -p sandybridge

    #! sbatch directives end here (put any additional directives above this line)

    #! Notes:
    #! Charging is determined by core number*walltime.

    #! ############################################################
    #! Modify the settings below to specify the application's environment, location
    #! and launch method:

    #! Optionally modify the environment seen by the application
    #! (note that SLURM reproduces the environment at submission irrespective of ~/.bashrc):
    . /etc/profile.d/modules.sh                # Leave this line (enables the module command)
    module purge                               # Removes all modules still loaded
    module load default-impi                   # REQUIRED - loads the basic environment

    #! Set up python
    echo -e "Running python: `which python`"
    . $HOME/arlenv/bin/activate
    export PYTHONPATH=$PYTHONPATH:$ARL
    echo "PYTHONPATH is ${PYTHONPATH}"
    module load python
    echo -e "Running python: `which python`"
    echo -e "Running dask-scheduler: `which dask-scheduler`"

    #! Work directory (i.e. where the job will run):
    workdir="$SLURM_SUBMIT_DIR"  # The value of SLURM_SUBMIT_DIR sets workdir to the directory
                                 # in which sbatch is run.

    #! Are you using OpenMP (NB this is unrelated to OpenMPI)? If so increase this
    #! safe value to no more than 16:
    export OMP_NUM_THREADS=1

    #CMD="jupyter nbconvert --execute --ExecutePreprocessor.timeout=3600 --to rst simple-dask.ipynb"
    #CMD="python dask_minimal.py"
    CMD="python3 imaging-distributed.py"

    cd $workdir
    echo -e "Changed directory to `pwd`.\n"

    JOBID=$SLURM_JOB_ID

    if [ "$SLURM_JOB_NODELIST" ]; then
            #! Create a hostfile:
            export NODEFILE=`generate_pbs_nodefile`
            cat $NODEFILE | uniq > hostfile.$JOBID
            echo -e "\nNodes allocated:\n================"
            echo `cat hostfile.$JOBID | sed -e 's/\..*$//g'`
    fi


    echo -e "JobID: $JOBID\n======"
    echo "Time: `date`"
    echo "Master node: `hostname`"
    echo "Current directory: `pwd`"

    # dask-worker --preload distributed_setup.py $scheduler &
    scheduler="`hostname`:8786"
    echo "About to dask-ssh on:"
    cat hostfile.$JOBID

    #! dask-ssh related options:
    #!  --nthreads INTEGER        Number of threads per worker process. Defaults to
    #!                            number of cores divided by the number of processes
    #!                            per host.
    #!  --nprocs INTEGER          Number of worker processes per host.  Defaults to
    #!                            one.
    #!  --hostfile PATH           Textfile with hostnames/IP addresses
    #!
    dask-ssh --nprocs 2 --nthreads 1 --scheduler-port 8786 --log-directory `pwd` --hostfile hostfile.$JOBID &
    sleep 10

    #! We need to tell dask Client (inside python) where the scheduler is running
    scheduler="`hostname`:8786"
    echo "Scheduler is running at ${scheduler}"
    export ARL_DASK_SCHEDULER=${scheduler}

    echo "About to execute $CMD"

    eval $CMD

    #! Wait for dash-ssh to be shutdown from the python
    wait %1

In the command CMD remember to shutdown the Client so the batch script will close the background dask-ssh and then exit.

Thw diagnostic pages can be tunneled. ARL emits the URL of the diagnostic page. For example::

      http://10.143.1.25:8787

Then to tunnel the pages::

      ssh hpccorn1@login.hpc.cam.ac.uk -L8080:10.143.1.25:8787

The diagnostic page is available from your local browser at::

      127.0.0.1:8080

