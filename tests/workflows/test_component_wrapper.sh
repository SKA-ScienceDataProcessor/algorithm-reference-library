#!/usr/bin/env bash
cd $ARL/tests/workflows
python $ARL/workflows/wrappers/component_wrapper.py --config test_create_vislist.json
python $ARL/workflows/wrappers/component_wrapper.py --config test_create_skymodel.json
python $ARL/workflows/wrappers/component_wrapper.py --config test_predict_vislist.json
python $ARL/workflows/wrappers/component_wrapper.py --config test_continuum_imaging.json
#python $ARL/workflows/wrappers/component_wrapper.py --config test_corrupt_vislist.json
#python $ARL/workflows/wrappers/component_wrapper.py --config test_ical.json