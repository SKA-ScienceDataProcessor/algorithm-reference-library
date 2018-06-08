"""

"""
from processing_components.component_support.arlexecute import arlexecute

from workflows.wrappers.execution_wrappers import initialise_config_wrapper, initialise_execution_wrapper
from workflows.wrappers.processing_wrappers import create_vislist_wrapper, continuum_imaging_wrapper, \
    create_skymodel_wrapper, simulate_wrapper

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Simulate a visibility list')
    parser.add_argument('--config', type=str, help='JSON configuration file')
    
    # Get the configuration definition, checking for validity
    config = initialise_config_wrapper(parser.parse_args().config)
    
    # Initialise execution framework and set up the logging
    initialise_execution_wrapper(config)
    
    # Define pipeline processing: If Dask is being used this creates a graph
    # otherwise it executes immediately
    
    assert config["component"]["framework"] == "ARL", "JSON specifies non-ARL component"
    
    arl_component = config["component"]["name"]
    if arl_component == "create_vislist":
        result = create_vislist_wrapper(config)
    
    elif arl_component == "create_skymodel":
        result = create_skymodel_wrapper(config)
    
    elif arl_component == "simulate":
        result = simulate_wrapper(config)
    
    elif arl_component == "continuum_imaging":
        result = continuum_imaging_wrapper(config)
    
    else:
        raise RuntimeError("%s is not wrapped" % arl_component)
    
    arlexecute.compute(result, sync=True)
    
    arlexecute.close()
