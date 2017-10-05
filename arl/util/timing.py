"""
See comment by Matt Alcock at http://stackoverflow.com/questions/1622943/timeit-versus-timing-decorator
"""

from functools import wraps
from time import time


def timing(f, verbose=False):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        if verbose:
            print('func:%r args:[%r, %r] took: %2.4f sec' % (f.__name__, args, kw, te - ts))
        else:
            print('func:%r took: %2.4f sec' % (f.__name__, te - ts))
        return result
    return wrap
