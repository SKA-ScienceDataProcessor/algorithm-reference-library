#!/usr/bin/env bash
mpiexec -n $1 -f machinefile python $2
