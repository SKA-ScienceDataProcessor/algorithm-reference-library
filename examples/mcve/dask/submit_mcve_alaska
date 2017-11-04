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
#SBATCH --nodes=3
#! How many (MPI) tasks will there be in total? (<= nodes*16)
#SBATCH --ntasks=3
#! Memory limit
##SBATCH --mem 63900
#! How much wallclock time will be required?
#SBATCH --time=00:10:00
#! What types of email messages do you wish to receive?
#SBATCH --mail-type=FAIL
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue
#! Do not change:
#SBATCH -p compute
#! What types of email messages do you wish to receive?
#SBATCH --mail-type=FAIL
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue

#! Modify the settings below to specify the application's environment, location
#! and launch method:

#! Optionally modify the environment seen by the application
#! (note that SLURM reproduces the environment at submission irrespective of ~/.bashrc):
module purge                               # Removes all modules still loaded

#! Set up python
. $HOME/arlenv/bin/activate
export PYTHONPATH=$PYTHONPATH:$ARL
echo "PYTHONPATH is ${PYTHONPATH}"

echo -e "Running python: `which python`"
echo -e "Running dask-scheduler: `which dask-scheduler`"

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
    ssh $host dask-worker --host ${host} --nprocs 1 --nthreads 1  \
    --memory-limit 0.1 --local-directory /tmp $scheduler:8786 &
        sleep 1
    hostIndex="1"
done
echo "Scheduler and workers now running"

CMD="python ./losing_workers-loop.py ${scheduler}:8786 > losing_workers_${JOBID}.out"
echo "About to execute $CMD"

eval $CMD

# Archive the results
archive="output_${JOBID}"
echo "Moving results to ${archive}"
mkdir ${archive}
mv "slurm-${JOBID}".out ${archive}
mv hostfile.${JOBID} ${archive}
mv dask-ssh* ${archive}
cp *.py ${archive}
cp ${0}  ${archive}
mv losing_workers_${JOBID}.out  ${archive}
