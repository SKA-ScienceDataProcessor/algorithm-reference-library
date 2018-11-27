#!/usr/bin/env bash
# NUMBER_NODES NUMBER_PROCS_PER_NODE NUMBER_FREQUENCY_WINDOWS NUMBER_THREADS MEMORY
python make_pipelines-timings.py  csd3 16 8 16
python make_pipelines-timings.py  csd3 16 8 32
python make_pipelines-timings.py  csd3 32 8 32
python make_pipelines-timings.py  csd3 32 8 64 
python make_pipelines-timings.py  csd3 32 16 512
python make_pipelines-timings.py  csd3 32 8 25
python make_pipelines-timings.py  csd3 32 4 64
python make_pipelines-timings.py  csd3 32 2 64
python make_pipelines-timings.py  csd3 32 1 32
python make_pipelines-timings.py  csd3 64 1 64 1 384



