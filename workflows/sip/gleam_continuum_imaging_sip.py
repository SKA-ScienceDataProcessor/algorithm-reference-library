import os
import sys

sys.path.append(os.path.join('..', '..'))

from processing_components.component_support.arlexecute import arlexecute

from workflows.wrappers.execution_wrappers import initialise_config_wrapper, define_execution_wrapper
from workflows.wrappers.processing_wrappers import continuum_imaging_component_wrapper

import logging

if __name__ == '__main__':
    log = logging.getLogger()
    
    # Get the configuration definition
    config = initialise_config_wrapper('gleam_sip.json')
    
    # Initialise execution framework and set up the logging
    define_execution_wrapper(config)
    
    # Define pipeline processing
    continuum_imaging_list = continuum_imaging_component_wrapper(config)
    
    log.info('About to run continuum imaging')
    arlexecute.compute(continuum_imaging_list, sync=True)
    
    arlexecute.close()
