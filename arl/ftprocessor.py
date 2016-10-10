# Tim Cornwell <realtimcornwell@gmail.com>
#
"""
Functions that aid fourier transform processing.
"""

import numpy

from astropy import constants as const

from arl.parameters import get_parameter
from arl.image_operations import Image, checkwcs, create_empty_image_like
from arl.synthesis_support import anti_aliasing_function

import logging

log = logging.getLogger("arl.ftpredict")


class ftpredict_base():
    """Class containing the functions required for Fourier processing

    ftpredict wraps functions for performing predict and image via iteration
    """
    
    def __init__(self, model, vis, params={}):
        """Initialise for processing

        :param model: Model image
        :param vis: visibility
        :param params: parameters for processing
        """
        self.field_of_view = get_parameter(params, 'field_of_view', 0.01)
        self.uvmax = get_parameter(params, 'uvmax', 1000.0)
        self.model = model
        self.vis = vis
        self.kernel = get_parameter(params, 'kernel', None)
        self.calls = 0
        self.maxcalls = 1
        self.params = params
        
        log.debug("ftpredict.base: ftpredict initialised")
    
    def __iter__(self):
        return self
    
    def __next__(self):
        """Get the next piece and process it?"""
        log.debug("ftprocessor_base: predicting the visibilities")
        yield self
    
    def __exit__(self):
        """Apply exit conditions for processing"""
        pass
    
class ftinvert_base():
    """Class containing the functions required for Fourier processing

    ftpredict wraps functions for performing predict and image via iteration
    """
    
    def __init__(self, model, vis, params={}):
        """Initialise for processing

        :param model: Model image
        :param vis: visibility
        """
        self.field_of_view = get_parameter('field_of_view', 0.01)
        self.uvmax = get_parameter(params, 'uvmax', 1000.0)
        self.model = model
        self.visibility = vis
        self.kernel = get_parameter(params, 'kernel', None)
        self.calls = 0
        self.maxcalls = 1
        self.params = params
        
        self.dirty = create_empty_image_like(model)
        self.dirty = create_empty_image_like(model)
        self.sumweights = 0.0
        
        log.debug("ftinvert.base: ftpredict initialised")
    
    def __iter__(self):
        return self
    
    def __next__(self):
        """Get the next?"""
        if self.calls == 0:
            self.calls += 1
            log.debug("ftpredict.next: Getting first set of data")
            return True
        elif self.calls < self.maxcalls:
            self.calls += 1
            log.debug("ftinvert.next: Getting data set %d" % self.calls)
            return True
        else:
            log.debug("ftinvert.next: data processed")
            return False

    def __exit__(self):
        """Apply exit conditions for processing"""
        pass

    def invert(self):
        """Invert using the appropriate subsections of the data and skymodel

        :return: Dirty image, PSF, sum of weights
        """
        log.debug("ftinvert.invert")
        
        pass
    
    def insert_images(self, dirty, psf, sumweights):
        """Insert the dirty image, psf, and sumweights
        
        These will be normalised on exit"""
        self.dirty.data += sumweights * dirty.data
        self.psf.data += sumweights * psf.data
        self.sumweights += sumweights
        
    def extract_vis(self):
        pass
    
    def weight(self):
        """Weight the visibility data"""
        log.debug("ftpredict.weight")
        return self.weight


class ftpredict_2d(ftpredict_base):
    """Simple 2D imaging with PSWF gridding function
    
    """
    
    def __init__(self, model, vis, params={}):
        super(ftpredict_2d, self).__init__(model, vis, params)
        log.debug("ftpredict_2d: Two-dimensional ftpredict initialised")
        self.kernelshape = [512, 512]
        self.kernel = anti_aliasing_function(self.kernelshape)


class ftpredict_image_partition(ftpredict_base):
    """Divide the image plane into facets and process each of those in turn"""
    
    def __init__(self, model, vis, params={}):
        super(ftpredict_image_partition, self).__init__(model, vis, params)
        self.nfacets = get_parameter(self.params, 'nfacets', default=3)
        self.maxcalls = self.nfacets ** 2
        log.debug("ftpredict_image_partition: ftpredict initialised")
        log.debug('ftpredict_image_partition: Processing %d image partitions' % (self.nfacets ** 2))
    
    def _getslice(self):
        """Get the slice for this call

        """
        dx = self.model.shape[3] // self.nfacets
        dy = self.model.shape[2] // self.nfacets
        x = self.calls // self.nfacets
        y = self.calls - x * self.nfacets
        x *= dx
        y *= dy
        return (..., slice(y - dy / 2.0, y + dy / 2.0), slice(x - dx / 2.0, x + dx / 2.0))
    
    def _extract_model(self):
        """Extract the current facet

        """
        sl = self._getslice()
        model = Image()
        model.data = self.model[sl]
        model.wcs = self.wcs.slice(sl)
        return model
    
    def __exit__(self):
        self.dirty = self.dirty / self.sumweights
        self.psf = self.psf / self.sumweights


class ftinvert_image_partition(ftinvert_base):
    """Divide the image plane into facets and process each of those in turn"""
    
    def __init__(self, model, vis, params={}):
        super(ftinvert_image_partition, self).__init__(model, vis, params)
        self.nfacets = get_parameter(self.params, 'nfacets', default=3)
        self.maxcalls = self.nfacets ** 2
        log.debug("ftpredict_image_partition: ftpredict initialised")
        log.debug('ftpredict_image_partition: Processing %d image partitions' % (self.nfacets ** 2))
    
    def _getslice(self):
        """Get the slice for this call

        """
        dx = self.model.shape[3] // self.nfacets
        dy = self.model.shape[2] // self.nfacets
        x = self.calls // self.nfacets
        y = self.calls - x * self.nfacets
        x *= dx
        y *= dy
        return (..., slice(y - dy / 2.0, y + dy / 2.0), slice(x - dx / 2.0, x + dx / 2.0))
    
    def insert_images(self, ims):
        """Insert the current images

        """
        dirty, psf, sumweights = ims
        sl = self._getslice()
        checkwcs(dirty.wcs, self.wcs.slice(sl))
        
        self.dirty.data[sl] += dirty.data[sl]
        self.psf.data[sl] += psf.data[sl]
        self.sumweights += sumweights
        
        return True


class ftpredict_fourier_partition(ftpredict_base):
    """Divide the image plane into facets and process each of those in turn"""
    
    def __init__(self, model, vis, params={}):
        super(ftpredict_fourier_partition, self).__init__(model, vis, params)
        self.nfacets = get_parameter(self.params, 'n_fourier_facets', default=1)
        self.maxcalls = self.nfacets ** 2
        log.debug("ftpredict_fourier_partition: ftpredict initialised")
        log.debug('ftpredict_fourier_partition: Processing %d image partitions' % (self.nfacets ** 2))
    
    def _getslice(self):
        """Get the slice for this call

        """
        dx = self.model.shape[3] // self.nfacets
        dy = self.model.shape[2] // self.nfacets
        x = self.calls // self.nfacets
        y = self.calls - x * self.nfacets
        x *= dx
        y *= dy
        return (..., slice(y - dy / 2.0, y + dy / 2.0), slice(x - dx / 2.0, x + dx / 2.0))
    
    def _extract_model(self):
        """Extract the current facet

        """
        sl = self._getslice()
        model = Image()
        model.data = self.model[sl]
        model.wcs = self.wcs.slice(sl)
        return model
    
    def __exit__(self):
        self.dirty = self.dirty / self.sumweights
        self.psf = self.psf / self.sumweights


class ftinvert_fourier_partition(ftinvert_base):
    """Divide the image plane into facets and process each of those in turn"""
    
    def __init__(self, model, vis, params={}):
        super(ftinvert_fourier_partition, self).__init__(model, vis, params)
        self.nfacets = get_parameter(self.params, 'n_fourier_facets', default=1)
        self.maxcalls = self.nfacets ** 2
        log.debug("ftpredict_fourier_partition: ftpredict initialised")
        log.debug('ftpredict_fourier_partition: Processing %d fourier partitions' % (self.nfacets ** 2))
    
    def _getslice(self):
        """Get the slice for this call

        """
        dx = self.model.shape[3] // self.nfacets
        dy = self.model.shape[2] // self.nfacets
        x = self.calls // self.nfacets
        y = self.calls - x * self.nfacets
        x *= dx
        y *= dy
        return (..., slice(y - dy / 2.0, y + dy / 2.0), slice(x - dx / 2.0, x + dx / 2.0))
    
    def insert_images(self, ims):
        """Insert the current images

        """
        dirty, psf, sumweights = ims
        sl = self._getslice()
        checkwcs(dirty.wcs, self.wcs.slice(sl))
        
        self.dirty.data[sl] += dirty.data[sl]
        self.psf.data[sl] += psf.data[sl]
        self.sumweights += sumweights
        
        return True


class ftpredict_wprojection(ftpredict_base):
    """W projection imaging
    
    """
    
    def __init__(self, model, vis, params={}):
        super(ftpredict_wprojection, self).__init__(model, vis, params)
        log.debug("ftpredict_wprojection: ftpredict initialised")
        self.wstep = get_parameter(self.params, "wstep", 10000.0)
        self.kernel = get_parameter(self.params, "wprojection_kernel", None)
        if self.kernel == None:
            log.debug("ftpredict_wprojection: constructing kernel function")
        else:
            log.debug("ftpredict_wprojection: using specified kernel function")


class ftinvert_wprojection(ftpredict_base):
    """W projection imaging

    """
    
    def __init__(self, model, vis, params={}):
        super(ftinvert_wprojection, self).__init__(model, vis, params)
        log.debug("ftpredict_wprojection: ftpredict initialised")
        self.wstep = get_parameter(self.params, "wstep", 10000.0)
        self.kernel = get_parameter(self.params, "kernel", None)
        if self.kernel == None:
            log.debug("ftpredict_wprojection: constructing kernel function")
        else:
            log.debug("ftpredict_wprojection: using specified kernel function")
