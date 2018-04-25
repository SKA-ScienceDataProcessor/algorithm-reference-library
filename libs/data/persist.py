"""
Persist objects on disk
"""
import cloudpickle


def arl_dump(obj, name: str):
    """Pickle an object and save to disk
    """
    with open(name, 'wb') as f:
        cloudpickle.dump(obj, f)


def arl_load(name: str):
    """Load an object from a pre-existing pickle
    """
    with open(name, 'rb') as f:
        return cloudpickle.load(f)
