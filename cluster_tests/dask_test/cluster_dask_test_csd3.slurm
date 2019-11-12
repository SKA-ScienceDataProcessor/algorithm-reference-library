#!/bin/bash
#!
#! Dask job script for CSD3
#! Tim Cornwell
#!

#!#############################################################
#!#### Modify the options in this section as appropriate ######
#!#############################################################

#! sbatch directives begin here ###############################
#! Name of the job:
#SBATCH -J CLUSTER_TEST
#! Which project should be charged:
#SBATCH -A SKA-SDP-SL2-CPU
#! How many whole nodes should be allocated?
#SBATCH --nodes=4
#! How many (MPI) tasks will there be in total? (<= nodes*16)
#SBATCH --ntasks=16
#! Memory limit: P3 has roughly 107GB per node
#SBATCH --mem 100000
#! How much wallclock time will be required?
#SBATCH --time=01:00:00
#! What types of email messages do you wish to receive?
#SBATCH --mail-type=FAIL,END
#! Where to send email messages
#SBATCH --mail-user=realtimcornwell@gmail.com
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue
#! Do not change:
#SBATCH -p skylake
#! What types of email messages do you wish to receivewhich?
#SBATCH --mail-type=FAIL
#! Ask for exclusive access
#SBATCH --exclusive
#! Same switch
#SBATCH --switches=1
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue
#! Optionally modify the environment seen by the application
#! (note that SLURM reproduces the environment at submission irrespective of ~/.bashrc):
module purge                               # Removes all modules still loaded

#! Set up python
echo -e "Running python: `which python`"
echo -e "Running dask-scheduler: `which dask-scheduler`"

module load slurm

cd $SLURM_SUBMIT_DIR
echo -e "Changed directory to `pwd`.\n"

JOBID=${SLURM_JOB_ID}
echo ${SLURM_JOB_NODELIST}

#! Create a hostfile:
scontrol show hostnames $SLURM_JOB_NODELIST | uniq > hostfile.$JOBID

scheduler=$(head -1 hostfile.$JOBID)
hostIndex=0
for host in `cat hostfile.$JOBID`; do
    echo "Working on $host ...."
    if [ "$hostIndex" = "0" ]; then
        echo "run dask-scheduler"
        ssh $host dask-scheduler --port=8786 &
        sleep 5
    fi
    echo "run dask-worker"
    ssh $host dask-worker --nprocs 4 --nthreads 1 \
    --memory-limit 10GB $scheduler:8786  &
        sleep 1
    hostIndex="1"
done
echo "Scheduler and workers now running"

echo "Scheduler is running at ${scheduler}:8786"

CMD="python cluster_dask_test.py ${scheduler}:8786 | tee cluster_dask_test.log"

eval $CMD

