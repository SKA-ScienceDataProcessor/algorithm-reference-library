# Tim Cornwell <realtimcornwell@gmail.com>
#
# Visibility data structure: a Table with columns ['uvw', 'time', 'antenna1', 'antenna2', 'vis', 'weight']
# and an attached attribute which is the frequency of each channel

import profile
import copy

from astropy import constants as const
from astropy.coordinates import SkyCoord, CartesianRepresentation
from astropy.table import Table, vstack

from crocodile.simulate import *

from arl.data_models import *
from arl.parameters import *

import logging

log = logging.getLogger("arl.visibility_iterators")


class vis_iterator_base():
    def __init__(self, vis, params):
        """Initialise the iterator
        """
        self.maximum_index = 10
        self.index = 0
        self.params = params
    
    def __iter__(self):
        """ Return the iterator itself
        """
        return self
    
    def __next__(self):
        try:
            result = self.vis.select(self.index)
        except IndexError:
            raise StopIteration
        self.index += 1
        return result


class vis_snapshot_iter(vis_iterator_base):
    def __init__(self, vis, params):
        """Initialise the iterator
        """
        super(vis_iterator_base, self).init(vis, params)
        self.maximum_index = 10
        self.index = 0
        self.params = params
    pass


class vis_wplane_iter(vis_iterator_base):
    pass


class vis_frequency_iter(vis_iterator_base):
    pass


class vis_parallactic_angle_iter(vis_iterator_base):
    pass
