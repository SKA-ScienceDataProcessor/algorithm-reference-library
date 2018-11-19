#! /bin/bash
# skylake nodes have 6GB per CPU and 12 cores per node
# skylake-himem have 12GB per CPU and 32 CPUs per node
TEMPLATE=slurm_submit.peta4-skylake.template
OUTPUT=slurm_job.peta4
NUMNODES_LIST='2 4 8 16'
NODETYPE_LIST='skylake skylake-himem'
NFREQ_LIST='21 41 71 101 203'

JOB_FOLDER=../../scripts/csd3-slurm/
WORK_DIRECTORY=../../workflows/mpi

for NUMNODES in $NUMNODES_LIST 
	do
	for NFREQ in $NFREQ_LIST 
		do 
		for NODETYPE in $NODETYPE_LIST 
			do
			# echo $NUMNODES $NFREQ $NODETYPE
			if [ $NODETYPE == 'skylake' ] 
				then 
				NUMCORES_LIST='6 12'
			elif [ $NODETYPE == 'skylake-himem' ] 
				then
				NUMCORES_LIST='16 32'
			else 
				echo 'Wrong nodetype'
			fi

			for NUMCORES in $NUMCORES_LIST 
				do
				NUMTASKS=$((NUMNODES*NUMCORES))
				#sed -e "s/@@NUMNODES@@/$NUMNODES/" -e "s/@@NUMTASKS@@/$NUMTASKS/" -e "s/@@NFREQ@@/$NFREQ/" -e "s/@@NODETYPE@@/$NODETYPE/" $TEMPLATE > $OUTPUT.$NODETYPE.$NUMNODES.$NUMTASKS.$NFREQ
				echo 'Submiting' $OUTPUT.$NODETYPE.$NUMNODES.$NUMTASKS.$NFREQ
				# sbatch $OUTPUT.$NODETYPE.$NUMNODES.$NUMTASKS.$NFREQ
				# rm $OUTPUT.$NODETYPE.$NUMNODES.$NUMTASKS.$NFREQ
			done
		done
	done
done


