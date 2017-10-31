#!/usr/bin/env bash
# NUMBER_NODES NUMBER_PROCS_PER_NODE NUMBER_TASKS NUMBER_FREQUENCY_WINDOWS NUMBER_THREADS EXECUTION_TIME
python make_imaging_pipelines.py  darwin 1 4 16 1
python make_imaging_pipelines.py  darwin 2 4 16 1
python make_imaging_pipelines.py  darwin 4 4 16 1
python make_imaging_pipelines.py  darwin 8 4 16 1
python make_imaging_pipelines.py  darwin 16 4 16 1
python make_imaging_pipelines.py  darwin 32 4 16 1
python make_imaging_pipelines.py  darwin 64 4 16 1