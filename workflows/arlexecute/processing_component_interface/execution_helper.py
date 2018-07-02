""" Wrappers needed for execution
"""

import logging

from json import loads

from ..execution_support.arlexecute import arlexecute
from workflows.arlexecute.processing_component_interface.arl_json.json_assertions import assert_valid_schema
from data_models.parameters import arl_path

def initialise_config_wrapper(config_file):
    """Obtain the configuration from a JSON file, validating against arl_schema
    
    :param config_file: Name of file containing JSON configuration
    :return: configuration
    """
    with open(config_file, 'r') as file:
        config = loads(file.read())
    
    assert_valid_schema(config, arl_path('workflows/arlexecute/processing_component_interface/arl_json/arl_schema'
                                         '.json'))
    
    return config


def initialise_logging_wrapper(conf):
    """ Initialise logging from JSON configuration
    
    See arl_schema.json

    :param conf: JSON configuratiion
    """
    if conf['logging']['level'] == "INFO":
        level = logging.INFO
    else:
        level = logging.DEBUG
        
    logging.basicConfig(filename=conf["buffer"]["directory"]+conf['logging']['filename'],
                        filemode=conf['logging']['filemode'],
                        format=conf['logging']['format'],
                        datefmt=conf['logging']['datefmt'],
                        level=level)


def initialise_execution_wrapper(conf):
    """Initialise the execution framework from JSON configuration
    
    See arl_schema.json
    
    :param conf: JSON configuratiion
    """
    arlexecute.set_client(use_dask=conf["execute"]["use_dask"],
                          n_workers=conf["execute"]["n_workers"],
                          memory_limit=conf["execute"]["memory_limit"])
    arlexecute.run(initialise_logging_wrapper, conf)
