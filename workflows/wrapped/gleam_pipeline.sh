#!/usr/bin/env bash
cd ${ARL}/workflows/wrapped
comp_location=${ARL}/processing_components/processing_component_interface
python ${comp_location}/processing_component_interface.py --config gleam_create_vislist.json
python ${comp_location}/processing_component_interface.py --config gleam_create_skymodel.json
python ${comp_location}/processing_component_interface.py --config gleam_predict_vislist.json
python ${comp_location}/processing_component_interface.py --config gleam_continuum_imaging.json
