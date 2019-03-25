#!/usr/bin/env bash
nthreads=1
memory=384
nprocs_per_node=16
context=timeslice
serial=True
rmax=1200
for nfreqwin in 64 128 256 512
    do
        filled=`expr ${nfreqwin} / ${nprocs_per_node}`
        for nnodes in ${filled} `expr ${filled} / 2` `expr ${filled} / 4` `expr ${filled} / 8`
            do
                # NUMBER_NODES NUMBER_PROCS_PER_NODE NUMBER_FREQUENCY_WINDOWS CONTEXT NUMBER_THREADS MEMORY SERIAL
                python make_pipelines-timings.py csd3 ${nnodes} ${nprocs_per_node} ${nfreqwin} ${context} \
                ${nthreads} ${memory} ${serial} ${rmax}
            done
    done
