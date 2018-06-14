""" This is the processing wrapper interface into the ARL, allowing accessing to wrapped components.

"""

import logging

from processing_components.component_support.arlexecute import arlexecute
from processing_components.processing_component_interface.execution_helper import initialise_config_wrapper, initialise_execution_wrapper

# Add new wrapped components here. These are accessed using globals()

from processing_components.processing_component_interface.processing_component_wrappers import create_vislist_wrapper, \
    create_skymodel_wrapper, predict_vislist_wrapper, corrupt_vislist_wrapper, continuum_imaging_wrapper

def component_wrapper(config):
    """Run an ARL component as described in a JSON dict or file
    
    :param config: JSON dict or file
    :return:
    """
    if isinstance(config, str):
        print('component_wrapper: read configuration from %s' % config)
        config = initialise_config_wrapper(config)

    elif isinstance(config, dict):
        print('component_wrapper: read configuration from dictionary')
    else:
        raise ValueError("config must be either a string of a JSON file or a dictionary")
    
    # Initialise execution framework and set up the logging
    initialise_execution_wrapper(config)
    
    log = logging.getLogger()
    
    assert config["component"]["framework"] == "ARL", "JSON specifies non-ARL component" % \
                                                      config["component"]["framework"]
    
    arl_component = config["component"]["name"]
    
    wrapper = arl_component + "_wrapper"
    assert wrapper in globals().keys(), 'ARL component %s is not wrapped' % arl_component
    
    log.info('component_wrapper: executing ARL component %s' % arl_component)
    print('component_wrapper: executing ARL component %s' % arl_component)
    
    result = globals()[wrapper](config)
    
    arlexecute.compute(result, sync=True)
    
    arlexecute.close()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Execute ARL componentst')
    parser.add_argument('--config', type=str, help='JSON configuration file')

    # Get the configuration definition, checking for validity
    config_file = parser.parse_args().config
    
    component_wrapper(config_file)
    
    
