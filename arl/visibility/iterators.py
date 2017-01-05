# Tim Cornwell <realtimcornwell@gmail.com>
#
""" Visibility iterators

"""

import numpy
import copy

from arl.data.parameters import get_parameter
from arl.data.data_models import Visibility
import logging
log = logging.getLogger("visibility.iterators")


class vis_iterator_base:
    def __init__(self, vis, params):
        """Initialise the iterator
        """
        self.vis = copy.copy(vis)
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
            log.debug("Index %i" % self.index)
            self.index += 1
        else:
            raise StopIteration
        return result


class vis_timeslice_iter(vis_iterator_base):
    def __init__(self, vis, params):
        
        """Initialise the iterator
        
        Assumes that the data are time ordered
        """
        # We have to make a copy or strange thing will happen!
        self.vis = copy.copy(vis)
        self.timeslice = get_parameter(params, "timeslice", 1.0)
        self.starttime = numpy.min(self.vis.time)
        self.stoptime = numpy.max(self.vis.time)
        self.timecursor = self.starttime
        super(vis_timeslice_iter, self).__init__(vis, params)

    def __next__(self):

        nrows = 0
        result = copy.copy(self.vis)
        while (nrows == 0) & (self.timecursor < self.stoptime):
            rows = ((self.vis.time >= (self.timecursor - self.timeslice / 2.0)) & \
                    (self.vis.time <  (self.timecursor + self.timeslice / 2.0)))
            nrows = numpy.sum(rows)
            self.timecursor += self.timeslice
            
        if nrows == 0:
            raise StopIteration

        return rows
