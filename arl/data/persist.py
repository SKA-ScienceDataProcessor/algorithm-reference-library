"""
Persist objects on disk
"""
import cloudpickle


def arl_dump(obj, name: str):
    """Pickle an object and save to disk
    """
    cloudpickle.dump(obj, open(name, 'wb'))


def arl_load(name: str):
    """Load an object from a pre-existing pickle
    """
    return cloudpickle.load(open(name, 'rb'))