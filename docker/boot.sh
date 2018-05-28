#!/usr/bin/env bash

source "${HOME}/.bash_profile"

# launch the notebook - the IP address to listen on is passed in via env-var IP
IP=${IP:-0.0.0.0}
NOTEBOOK_PORT=${NOTEBOOK_PORT:-8888}
export JUPYTER_PASSWORD=${JUPYTER_PASSWORD:-changeme}
jupyter notebook --allow-root --no-browser --ip=${IP} --port=${NOTEBOOK_PORT} /arl
#jupyter lab --allow-root --no-browser --ip=${IP} --port=${PORT} "$@" /arl/examples/arl/

# source activate dask-distributed
#jupyter lab --allow-root "$@"
