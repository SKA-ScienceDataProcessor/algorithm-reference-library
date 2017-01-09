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


class vis_timeslice_iter():
    def __init__(self, vis, params):
        
        """Initialise the iterator
        
        """
        # We have to make a copy or strange thing will happen!
        self.vis = copy.copy(vis)
        self.timeslice = get_parameter(params, "timeslice", 1.0)
        self.starttime = numpy.min(self.vis.time)
        self.stoptime = numpy.max(self.vis.time)
        self.timecursor = self.starttime

    def __iter__(self):
        """ Return the iterator itself
        """
        return self

    def __next__(self):

        nrows = 0
        while (nrows == 0) & (self.timecursor < self.stoptime):
            rows = ((self.vis.time >= (self.timecursor - self.timeslice / 2.0)) & \
                    (self.vis.time <  (self.timecursor + self.timeslice / 2.0)))
            nrows = numpy.sum(rows)
            self.timecursor += self.timeslice
            
        if nrows == 0:
            raise StopIteration

        return rows
