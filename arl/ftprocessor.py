# Tim Cornwell <realtimcornwell@gmail.com>
#
"""
Functions that aid fourier transform processing.
"""

from arl.parameters import get_parameter
from arl.image_operations import create_image_from_subsection
from arl.synthesis_support import anti_aliasing_function


class ftprocessor:
    """Base Class containing the functions required for Fourier processing

    """
    
    def __init__(self, field_of_view, uvmax, model, kernel=None, params={}):
        """Initialise for processing

        :param field_of_view: Field of view (radians)
        :param uvmax:(longest uvw)
        :param model: Image to be used as model
        :param kernel: Convolution kernel
        """
        self.field_of_view = None
        self.uvmax = None
        self.model = model
        self.kernel = kernel
        self.calls = 0
        self.params = params
    
    def next(self):
        self.calls += 1
        return self.calls < 2
    
    def do_grid(self, cimage, uv, vis, **kwargs):
        pass
    
    def do_degrid(self, cimage, uv, vis, **kwargs):
        pass
    
    def vis(self):
        return self.vis
    
    def wt(self):
        return self.weight
    
    def extract_image(self):
        return self.model
    
    def insert_image(self, model):
        self.model = model


class ftprocessor_2d(ftprocessor):
    """Simple 2D imaging with PSWF gridding function
    
    """
    
    def __init__(self, field_of_view, uvmax, model, kernel=None, params={}):
        super(ftprocessor_2d, self).__init__(field_of_view, uvmax, model, kernel, params)
        self.field_of_view = None
        self.uvmax = None
        self.kernel = anti_aliasing_function(self.model.shape)


class ftprocessor_facet(ftprocessor):
    """Faceted imaging"""
    
    def __init__(self, field_of_view, uvmax, model, kernel=None, params={}):
        super(ftprocessor_2d, self).__init__(field_of_view, uvmax, model, kernel, params)
        self.field_of_view = None
        self.uvmax = None
        self.nfacets = get_parameter(self.params, 'nfacets', default=3)
        self.kernel = anti_aliasing_function(self.model.shape)
    
    def image(self, index):
        x = index // self.nfacets
        y = index - x * self.nfacets
        dx = self.model.shape[3] // self.nfacets
        dy = self.model.shape[2] // self.nfacets
        x *= dx
        y *= dy
        xslice = [x - dx / 2.0, x + dx / 2.0]
        yslice = [y - dy / 2.0, x + dy / 2.0]
        return create_image_from_subsection(xslice, yslice)
    
    def next(self):
        self.calls += 1
        return self.calls < self.nfacets ** 2


class ftprocessor_wprojection(ftprocessor):
    """W projection imaging"""
    
    def __init__(self, field_of_view, uvmax, model, kernel=None, params={}):
        super(ftprocessor_2d, self).__init__(field_of_view, uvmax, model, kernel, params)
        self.field_of_view = None
        self.uvmax = None
        self.kernel = anti_aliasing_function(self.model.shape)
