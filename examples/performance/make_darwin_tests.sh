#!/usr/bin/env bash
# NUMBER_NODES NUMBER_PROCS_PER_NODE NUMBER_TASKS NUMBER_FREQUENCY_WINDOWS NUMBER_THREADS EXECUTION_TIME
python make_pipelines-timings.py  1 1 16 4
python make_pipelines-timings.py  2 1 16 4
python make_pipelines-timings.py  4 1 16 4
python make_pipelines-timings.py  8 1 16 4
python make_pipelines-timings.py  16 1 16 4
python make_pipelines-timings.py  32 1 16 4
python make_pipelines-timings.py  64 1 16 4