# Tim Cornwell <realtimcornwell@gmail.com>
#
""" Visibility iterators

"""

import numpy
import copy

from arl.data.parameters import get_parameter
import logging
log = logging.getLogger(__name__)


class vis_timeslice_iter():
    """ Time slice iterator
    
    If timeslice='auto' then timeslice is taken to be the difference between the first two
    unique elements of the vis time.
          
    :param timeslice: Timeslice (seconds) ('auto')
    :returns: Boolean array with selected rows=True
        
    """
    def __init__(self, vis, **kwargs):
        
        """Initialise the iterator
        
        """
        self.vis = vis
        self.uniquetimes = numpy.unique(vis.time)
        self.timeslice = get_parameter(kwargs, "timeslice", 'auto')
        if self.timeslice == 'auto':
            log.info('vis_timeslice_iter: Found %d unique times' % len(self.uniquetimes))
            if len(self.uniquetimes) > 1:
                self.timeslice = (self.uniquetimes[1] - self.uniquetimes[0])
                log.debug('vis_timeslice_auto: Guessing time interval to be %.2f s' % self.timeslice)
            else:
                # Doesn't matter what we set it to.
                self.timeslice = vis.integration_time[0]
            
        self.cursor = 0

    def __iter__(self):
        """ Return the iterator itself
        """
        return self

    def __next__(self):

        nrows = 0
        while (nrows == 0) & (self.cursor < len(self.uniquetimes)):
            self.timecursor = self.uniquetimes[self.cursor]
            rows = numpy.abs(self.vis.time - self.timecursor) < 0.001
            nrows = numpy.sum(rows)
            self.cursor += 1
            
        if nrows == 0:
            raise StopIteration

        return rows
