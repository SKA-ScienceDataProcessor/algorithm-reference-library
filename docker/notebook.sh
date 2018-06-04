#!/usr/bin/env bash

source "${HOME}/.bash_profile"
# source activate dask-distributed
jupyter lab --allow-root "$@"
