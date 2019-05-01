#! /bin/bash
# skylake nodes have 6GB per CPU and 12 cores per node
# skylake-himem have 12GB per CPU and 32 CPUs per node
# Script should be run from $ARLROOT/scripts/csd3-slurm

#TEMPLATE=slurm_submit.peta4-skylake.template
#OUTPUT=slurm_job.peta4
NUMNODES_LIST='2 4 8 16'
NODETYPE_LIST='skylake-himem'
NFREQ_LIST='21 41 71 101 203 407'

RESULTS_OUTPUT=mpitest

WORK_DIRECTORY=../../workflows/mpi

cd $WORK_DIRECTORY
#cd results/mpi
cd results/dask/some-results
echo 'Current directory' $PWD
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
				NUMCORES_LIST='16'
			else 
				echo 'Wrong nodetype'
			fi
			for NUMCORES in $NUMCORES_LIST 
				do
					# For MPI tests
				NUMTASKS=$((NUMNODES*NUMCORES))
				#sed -e "s/@@NUMNODES@@/$NUMNODES/" -e "s/@@NUMTASKS@@/$NUMTASKS/" -e "s/@@NFREQ@@/$NFREQ/" -e "s/@@NODETYPE@@/$NODETYPE/" $TEMPLATE > $OUTPUT.$NODETYPE.$NUMNODES.$NUMTASKS.$NFREQ
				#cat $RESULTS_OUTPUT-$NODETYPE-$NUMNODES-$NUMTASKS-$NFREQ.out |grep "0: predict finished in" | awk {'print $NUMNODES $NUMTASKS $NFREQ $5'}
				cat $RESULTS_OUTPUT-$NODETYPE-$NUMNODES-$NUMTASKS-$NFREQ.out |grep ^"0: predict finished in" | awk {"print ${NUMNODES} \"\t\" ${NUMTASKS} \"\t\" $NFREQ \"\t\" \$5"} >> predict-results.txt
				cat $RESULTS_OUTPUT-$NODETYPE-$NUMNODES-$NUMTASKS-$NFREQ.out |grep ^"0: invert finished in" | awk {"print ${NUMNODES} \"\t\" ${NUMTASKS} \"\t\" $NFREQ \"\t\" \$5"} >> invert-results.txt
				cat $RESULTS_OUTPUT-$NODETYPE-$NUMNODES-$NUMTASKS-$NFREQ.out |grep ^"0: continuum imaging finished in" | awk {"print ${NUMNODES} \"\t\" ${NUMTASKS} \"\t\" $NFREQ \"\t\" \$6"} >> contimg-results.txt
				cat $RESULTS_OUTPUT-$NODETYPE-$NUMNODES-$NUMTASKS-$NFREQ.out |grep ^"0: ical finished in" | awk {"print ${NUMNODES} \"\t\" ${NUMTASKS} \"\t\" $NFREQ \"\t\" \$5"} >> ical-results.txt
					# For Dask tests
				NUMTASKS=$NUMCORES
				cat $RESULTS_OUTPUT-$NODETYPE-$NUMNODES-$NUMTASKS-$NFREQ.out |grep ^"predict finished in" | awk {"print ${NUMNODES} \"\t\" ${NUMTASKS} \"\t\" $NFREQ \"\t\" \$4"} >> predict-results.txt
				cat $RESULTS_OUTPUT-$NODETYPE-$NUMNODES-$NUMTASKS-$NFREQ.out |grep ^"invert finished in" | awk {"print ${NUMNODES} \"\t\" ${NUMTASKS} \"\t\" $NFREQ \"\t\" \$4"} >> invert-results.txt
				cat $RESULTS_OUTPUT-$NODETYPE-$NUMNODES-$NUMTASKS-$NFREQ.out |grep ^"continuum imaging finished in" | awk {"print ${NUMNODES} \"\t\" ${NUMTASKS} \"\t\" $NFREQ \"\t\" \$5"} >> contimg-results.txt
				cat $RESULTS_OUTPUT-$NODETYPE-$NUMNODES-$NUMTASKS-$NFREQ.out |grep ^"ical finished in" | awk {"print ${NUMNODES} \"\t\" ${NUMTASKS} \"\t\" $NFREQ \"\t\" \$4"} >> ical-results.txt

			done
		done
	done
done


