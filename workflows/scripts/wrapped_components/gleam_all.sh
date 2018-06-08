#!/usr/bin/env bash
cd $ARL/workflows/scripts/wrapped_components
python $ARL/workflows/wrappers/component_wrapper.py --config gleam_create_vislist.json
python $ARL/workflows/wrappers/component_wrapper.py --config gleam_create_skymodel.json
python $ARL/workflows/wrappers/component_wrapper.py --config gleam_predict_vislist.json
python $ARL/workflows/wrappers/component_wrapper.py --config gleam_continuum_imaging.json
#python $ARL/workflows/wrappers/component_wrapper.py --config gleam_corrupt_vislist.json
#python $ARL/workflows/wrappers/component_wrapper.py --config gleam_ical.json