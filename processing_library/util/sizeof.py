""" From https://goshippo.com/blog/measure-real-size-any-python-object/

"""

__all__ = ['get_size']

import sys

from distributed.protocol import pickle

def get_size(obj):
    return len(pickle.dumps(obj))
