#!/usr/bin/env bash
# HOSTNAME NUMBER_NODES NUMBER_PROCS_PER_NODE NUMBER_TASKS NUMBER_FREQUENCY_WINDOWS NUMBER_THREADS EXECUTION_TIME
python make_pipelines-timings.py  alaska 4 1 32 1
python make_pipelines-timings.py  alaska 4 2 32 1
python make_pipelines-timings.py  alaska 4 4 32 1
python make_pipelines-timings.py  alaska 4 8 32 1
python make_pipelines-timings.py  alaska 4 16 32 1
python make_pipelines-timings.py  alaska 4 32 32 1
