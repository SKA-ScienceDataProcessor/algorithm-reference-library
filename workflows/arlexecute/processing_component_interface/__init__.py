"""The python script processing_component_interface enables enable running ARL components from another environment
such as bash.

A bash script example for a continuum imaging pipeline is::

    #!/usr/bin/env bash
    cd ${ARL}/workflows/wrapped
    comp_location=${ARL}/processing_components/processing_component_interface
    python ${comp_location}/processing_component_interface.py --config gleam_create_vislist.json
    python ${comp_location}/processing_component_interface.py --config gleam_create_skymodel.json
    python ${comp_location}/processing_component_interface.py --config gleam_predict_vislist.json
    python ${comp_location}/processing_component_interface.py --config gleam_continuum_imaging.json

To be available in this way, a component must be wrapped appropriately and placed in
processing_component_wrappers.py. For example, here is a simple wrapper::

    def corrupt_vislist_wrapper(conf):
        vis_list = buffer_data_model_to_memory(conf["buffer"], conf['inputs']['vis_list'])
        phase_error = json_to_quantity(conf['corrupt_vislist']['phase_error']).to('rad').value
        
        corrupted_vislist = corrupt_workflow(vis_list,
                                              phase_error=phase_error,
                                              amplitude_error=conf['corrupt_vislist']['amplitude_error'])
        
        return arlexecute.execute(memory_data_model_to_buffer)(corrupted_vislist, conf["buffer"], conf['outputs']['vis_list'])


the wrapper for predict is::

    def predict_vislist_wrapper(conf):
        vis_list = buffer_data_model_to_memory(conf["buffer"], conf['inputs']['vis_list'])
        skymodel = buffer_data_model_to_memory(conf["buffer"], conf['inputs']['skymodel'])
        
        flux_limit = conf['primary_beam']['flux_limit']
        
        if conf["primary_beam"]["apply"]:
            def apply_pb_image(vt, model):
                telescope = vt.configuration.name
                pb = create_pb(model, telescope)
                model.data *= pb.data
                return model
            
            def apply_pb_comp(vt, model, comp):
                telescope = vt.configuration.name
                pb = create_pb(model, telescope)
                return apply_beam_to_skycomponent(comp, pb, flux_limit)
            
            image_list = [arlexecute.execute(apply_pb_image, nout=1)(v, skymodel.images[i]) for i, v in enumerate(vis_list)]
            if len(skymodel.components) > 1:
                component_list = [arlexecute.execute(apply_pb_comp, nout=1)(v, skymodel.images[i], skymodel.components)
                                for i, v in enumerate(vis_list)]
            else:
                component_list = []
        else:
            image_list = [skymodel.images[0] for v in vis_list]
            component_list = skymodel.components
        
        future_vis_list = arlexecute.scatter(vis_list)
        predicted_vis_list = [arlexecute.execute(predict_skycomponent_visibility)(v, component_list)
                              for v in future_vis_list]
        predicted_vis_list = predict_workflow(predicted_vis_list, image_list,
                                               context=conf['imaging']['context'],
                                               vis_slices=conf['imaging']['vis_slices'])
        
        return arlexecute.execute(memory_data_model_to_buffer)(predicted_vis_list, conf["buffer"],
                                                             conf["outputs"]["vis_list"])
                                                         
The JSON files contain the name of the component to be run and all the parameters necessary. An example of a JSON
file is::

    {
        "execute": {
            "use_dask": true,
            "n_workers": 4,
            "memory_limit": 4000000000
        },
        "component": {
            "framework": "ARL",
            "name": "predict_vislist"
        },
        "logging": {
            "filename": "test_pipeline.log",
            "filemode": "a",
            "format": "%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
            "datefmt": "%H:%M:%S",
            "level": "INFO"
        },
        "buffer": {
            "directory": "test_results/"
        },
    
        "inputs": {
            "skymodel": {
                "name":"test_skymodel.hdf",
                "data_model": "SkyModel"
            },
            "vis_list": {
                "name": "test_empty_vislist.hdf",
                "data_model": "BlockVisibility"
            }
        },
        "outputs": {
            "vis_list": {
                "name": "test_perfect_vislist.hdf",
                "data_model": "BlockVisibility"
            }
        },
        "imaging": {
            "context": "wstack",
            "vis_slices": 11
        },
        "primary_beam": {
            "apply": true,
            "flux_limit" : {"value": 0.01, "unit":"Jy"}
        }
    }

The parameters for the component are passed via a JSON file, either via python::
    
    component_wrapper("gleam_continuum_imaging.json")

or the dict derived from JSON may be passed directly::

    config = initialise_config_wrapper("gleam_continuum_imaging.json")
    component_wrapper(config)
    
or from bash::

    python component_wrapper -- config "gleam_continuum_imaging.json"
    

    
Examples of json files are in tests/workflows/.

"""