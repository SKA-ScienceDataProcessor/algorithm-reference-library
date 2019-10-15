#!/bin/bash

source /apps/BSCTOOLS/extrae/3.4.1/impi+libgomp4.9/etc/extrae.sh

${EXTRAE_HOME}/bin/mpi2prv -f TRACE.mpits -o mpi_ping.prv
