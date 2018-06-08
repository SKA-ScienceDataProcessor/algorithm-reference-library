""" Script to execute wrapped components

The parameters for the component are passed via a JSON file, the name of which is an argument::
    
    python component_wrapper -- config "gleam_continuum_imaging.json"
    
"""

import logging

from workflows.wrappers.execution_wrappers import initialise_config_wrapper, initialise_execution_wrapper
from workflows.wrappers.processing_wrappers import *

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Execute ARL componentst')
    parser.add_argument('--config', type=str, help='JSON configuration file')

    # Get the configuration definition, checking for validity
    config_file = parser.parse_args().config
    config = initialise_config_wrapper(config_file)
    
    # Initialise execution framework and set up the logging
    initialise_execution_wrapper(config)

    log = logging.getLogger()
    
    log.info('component_wrapper: read configuration from %s' % config_file)
    print('component_wrapper: read configuration from %s' % config_file)

    assert config["component"]["framework"] == "ARL", "JSON specifies non-ARL component"
    
    arl_component = config["component"]["name"]
    
    wrapper = arl_component+"_wrapper"
    assert wrapper in locals().keys(), 'ARL component %s is not wrapped' % arl_component
    
    log.info('component_wrapper: executing ARL component %s' % arl_component)
    print('component_wrapper: executing ARL component %s' % arl_component)

    result = locals()[wrapper](config)
    
    arlexecute.compute(result, sync=True)
    
    arlexecute.close()
