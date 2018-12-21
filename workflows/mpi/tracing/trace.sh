#!/bin/bash

echo $PWD
echo ${EXTRAE_HOME}
source ${EXTRAE_HOME}/etc/extrae.sh

export PYTHONPATH=${EXTRAE_HOME}/libexec:$PYTHONPATH
export EXTRAE_CONFIG_FILE=${ARLROOT}/workflows/mpi/tracing/extrae.xml
#export EXTRAE_CONFIG_FILE=./extrae.xml
export LD_PRELOAD=${EXTRAE_HOME}/lib/libmpitrace.so # For C apps
#export LD_PRELOAD=${EXTRAE_HOME}/lib/libmpitracef.so # For Fortran apps

## Run the desired program
$*

