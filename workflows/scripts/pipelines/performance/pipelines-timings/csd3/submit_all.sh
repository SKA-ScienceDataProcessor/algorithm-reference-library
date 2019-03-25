#!/usr/bin/env bash
while [[ $# -gt 0 ]]
do
    sbatch $1
    shift 1
done