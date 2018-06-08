"""Wrappers enable running ARL components from another environment such as bash.

A bash script example for a continuum imaging pipeline is::

    #!/usr/bin/env bash
    cd $ARL/workflows/scripts/wrapped_components
    python $ARL/workflows/wrappers/component_wrapper.py --config gleam_create_vislist.json
    python $ARL/workflows/wrappers/component_wrapper.py --config gleam_create_skymodel.json
    python $ARL/workflows/wrappers/component_wrapper.py --config gleam_predict_vislist.json
    python $ARL/workflows/wrappers/component_wrapper.py --config gleam_continuum_imaging.json

To be available in this way, a component must be wrapped appropriately and placed in processing_wrappers.py. For
example, the wrapper for predict is::

    def predict_vislist_wrapper(conf):
        vis_list = import_blockvisibility_from_hdf5(conf['inputs']['vis_list'])
        skymodel = import_skymodel_from_hdf5(conf['inputs']['skymodel'])
        
        if conf['applypb']:
            def apply_pb(vt, model):
                telescope = vt.configuration.name
                pb = create_pb(model, telescope)
                model.data *= pb.data
                return model
            
            image_list = [arlexecute.execute(apply_pb, nout=1)(v, skymodel.images[0]) for v in vis_list]
        else:
            image_list = [skymodel.images[0] for v in vis_list]
        
        future_vis_list = arlexecute.scatter(vis_list)
        predicted_vis_list = predict_component(future_vis_list, image_list,
                                               context=conf['imaging']['context'],
                                               vis_slices=conf['imaging']['vis_slices'])
        
        def output_vislist(v):
            return export_blockvisibility_to_hdf5(v, conf['outputs']['vis_list'])
        
        return arlexecute.execute(output_vislist)(predicted_vis_list)

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
            "filename": "/Users/timcornwell/Code/algorithm-reference-library/test_results/gleam-pipeline.log",
            "filemode": "a",
            "format": "%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
            "datefmt": "%H:%M:%S",
            "level": "INFO"
        },
        "inputs": {
            "skymodel": "/Users/timcornwell/Code/algorithm-reference-library/test_results/gleam_skymodel.hdf",
            "vis_list": "/Users/timcornwell/Code/algorithm-reference-library/test_results/gleam_empty_vislist.hdf"
        },
        "outputs": {
            "vis_list": "/Users/timcornwell/Code/algorithm-reference-library/test_results/gleam_perfect_vislist.hdf"
        },
        "imaging": {
            "context": "wstack",
            "vis_slices": 11
        },
        "applypb": true
    }
"""