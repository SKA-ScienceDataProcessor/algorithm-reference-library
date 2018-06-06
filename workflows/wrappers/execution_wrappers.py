
import logging

from json import loads

from processing_components.component_support.arlexecute import arlexecute
from wrappers.arl_json.json_assertions import assert_valid_schema
from data_models.parameters import arl_path

def initialise_config_wrapper(config_file):
    with open(config_file, 'r') as file:
        config = loads(file.read())
    
    assert_valid_schema(config, arl_path('workflows/wrappers/arl_json/arl_schema.json'))
    
    return config


def initialise_logging_wrapper(conf):
    if conf['logging']['level'] == "INFO":
        level = logging.INFO
    else:
        level = logging.DEBUG
    
    logging.basicConfig(filename=conf['logging']['filename'],
                        filemode=conf['logging']['filemode'],
                        format=conf['logging']['format'],
                        datefmt=conf['logging']['datefmt'],
                        level=level)


def define_execution_wrapper(conf):
    arlexecute.set_client(use_dask=conf["execute"]["use_dask"],
                          n_workers=conf["execute"]["n_workers"],
                          memory_limit=conf["execute"]["memory_limit"])
    arlexecute.run(initialise_logging_wrapper, conf)
