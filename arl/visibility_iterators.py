# Tim Cornwell <realtimcornwell@gmail.com>
#
# Visibility data structure: a Table with columns ['uvw', 'time', 'antenna1', 'antenna2', 'vis', 'weight']
# and an attached attribute which is the frequency of each channel

import profile
import copy

from astropy import constants as const
from astropy.coordinates import SkyCoord, CartesianRepresentation
from astropy.table import Table, vstack

from arl.coordinate_support import *

from arl.data_models import *
from arl.parameters import *

import logging

log = logging.getLogger("arl.visibility_iterators")


class vis_iterator_base():
    def __init__(self, vis, params):
        """Initialise the iterator
        """
        self.vis = vis
        self.maximum_index = 10
        self.index = 0
        self.params = params
    
    def __iter__(self):
        """ Return the iterator itself
        """
        return self
    
    def __next__(self):
        if self.index < self.maximum_index:
            result = self.vis
            log.debug("Index %i" % (self.index))
            self.index += 1
        else:
            raise StopIteration
        return result


class vis_snapshot_iter(vis_iterator_base):
    def __init__(self, vis, params):
        
        """Initialise the iterator
        """
        super(vis_iterator_base, self).__init__(vis, params)
        self.maximum_index = 10
        pass


class vis_wslice_iter(vis_iterator_base):
    def __init__(self, vis, params):
        """Initialise the iterator
        """
        super(vis_wslice_iter, self).__init__(vis, params)
        self.maximum_index = 10
        pass


class vis_frequency_iter(vis_iterator_base):
    def __init__(self, vis, params):
        """Initialise the iterator
        """
        super(vis_iterator_base, self).__init__(vis, params)
        self.maximum_index = 10
        pass


class vis_parallactic_angle_iter(vis_iterator_base):
    def __init__(self, vis, params):
        """Initialise the iterator
        """
        super(vis_iterator_base, self).__init__(vis, params)
        self.maximum_index = 10
        pass

