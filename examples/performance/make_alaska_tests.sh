#!/usr/bin/env bash
# NUMBER_NODES NUMBER_PROCS_PER_NODE NUMBER_TASKS NUMBER_FREQUENCY_WINDOWS NUMBER_THREADS EXECUTION_TIME
python make_pipelines-timings.py  alaska 1 1 16
python make_pipelines-timings.py  alaska 2 1 16
python make_pipelines-timings.py  alaska 3 1 16
python make_pipelines-timings.py  alaska 4 1 16

python make_pipelines-timings.py  alaska 1 2 16
python make_pipelines-timings.py  alaska 2 2 16
python make_pipelines-timings.py  alaska 3 2 16
python make_pipelines-timings.py  alaska 4 2 16

python make_pipelines-timings.py  alaska 1 4 16
python make_pipelines-timings.py  alaska 2 4 16
python make_pipelines-timings.py  alaska 3 4 16
python make_pipelines-timings.py  alaska 4 4 16

python make_pipelines-timings.py  alaska 1 8 16
python make_pipelines-timings.py  alaska 2 8 16
python make_pipelines-timings.py  alaska 3 8 16
python make_pipelines-timings.py  alaska 4 8 16

python make_pipelines-timings.py  alaska 1 16 16
python make_pipelines-timings.py  alaska 2 16 16
python make_pipelines-timings.py  alaska 3 16 16
python make_pipelines-timings.py  alaska 4 16 16

python make_pipelines-timings.py  alaska 1 32 16
python make_pipelines-timings.py  alaska 2 32 16
python make_pipelines-timings.py  alaska 3 32 16
python make_pipelines-timings.py  alaska 4 32 16

