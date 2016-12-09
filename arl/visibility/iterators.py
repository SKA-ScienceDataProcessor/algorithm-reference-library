# Tim Cornwell <realtimcornwell@gmail.com>
#
""" Visibility iterators

"""

import numpy
import copy

from arl.data.parameters import get_parameter
import logging
log = logging.getLogger("arl.visibility_iterators")


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


class vis_slice_iter(vis_iterator_base):
    """ General visibility iterator
    """
    def __init__(self, vis, params):
        
        """Initialise the iterator

        Assumes that the data are time ordered
        """
        # We have to make a copy or strange thing will happen!
        self.vis = copy.copy(vis)
        self.sliceaxis = get_parameter(params, "vissliceaxis", "time")
        self.slice = get_parameter(params, "visslice", 60.0)
        self.start = numpy.min(self.vis[self.sliceaxis])
        self.stop = numpy.max(self.vis[self.sliceaxis])
        self.cursor = self.start
        super(vis_slice_iter, self).__init__(vis, params)
    
    def __next__(self):
        
        nrows = 0
        rows = None
        result = copy.copy(self.vis)
        while (nrows == 0) & (self.cursor < self.stop):
            rows = ((self.vis[self.sliceaxis] >= (self.cursor - self.slice / 2.0)) & \
                    (self.vis[self.sliceaxis] < (self.cursor + self.slice / 2.0)))
            nrows = numpy.sum(rows)
            self.cursor += self.slice
        
        if nrows == 0 or rows is None:
            raise StopIteration
        else:
            result.data = self.vis.data[rows]
        
        return result


class vis_timeslice_iter(vis_iterator_base):
    def __init__(self, vis, params):
        
        """Initialise the iterator
        
        Assumes that the data are time ordered
        """
        # We have to make a copy or strange thing will happen!
        self.vis = copy.copy(vis)
        self.timeslice = get_parameter(params, "timeslice", 60.0)
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
        else:
            result.data = self.vis.data[rows]
            
        return result


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
        super(vis_frequency_iter, self).__init__(vis, params)
        self.maximum_index = 10
        pass


class vis_parallactic_angle_iter(vis_iterator_base):
    def __init__(self, vis, params):
        """Initialise the iterator
        """
        super(vis_parallactic_angle_iter, self).__init__(vis, params)
        self.maximum_index = 10
        pass

