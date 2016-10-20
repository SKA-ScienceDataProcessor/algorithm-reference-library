# Tim Cornwell <realtimcornwell@gmail.com>
#
""" Visibility iterators

"""

import logging
log = logging.getLogger("arl.visibility_iterators")


class vis_iterator_base:
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
            log.debug("Index %i" % self.index)
            self.index += 1
        else:
            raise StopIteration
        return result


class vis_snapshot_iter(vis_iterator_base):
    def __init__(self, vis, params):
        
        """Initialise the iterator
        """
        super(vis_snapshot_iter, self).__init__(vis, params)
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

