""" Generic functions converted to arlexecute components. arlexecute is a wrapper for`Dask <http://dask.pydata.org/>`_
is a python-based flexible parallel computing library for analytic computing. Dask.delayed can be used to wrap
functions for deferred execution thus allowing construction of graphs.

For example, consider a trivial example to take the square root of an image::

        def imagerooter(im, **kwargs):
            im.data_models = numpy.sqrt(numpy.abs(im.data_models))
            return im
        root = generic_image_arlexecute(imagerooter, myimage,  image_raster_iter, facets=4).compute()

We create the graph and execute it immediately. We are using the  image_raster_iter so the image will be divided into 16
subimages and passed to processing by imagerooter, and then the answers are reassembled.

We  could keep the graph and use it in other components. See the imaging-dask note book for more detail.
"""

from ..execution_support.arlexecute import arlexecute
from processing_components.image.gather_scatter import image_scatter_facets, image_gather_facets
from data_models.memory_data_models import Image
from libs.image.operations import copy_image, create_empty_image_like

def generic_image_iterator_arlexecute(imagefunction, im: Image, iterator, **kwargs):
    """ Definition of interface for generic_image_arlexecute
    
    This generates a graph for imagefunction. Note that im cannot be a graph itself.

    :func imagefunction: Function to be applied to all pixels
    :param im: Image to be processed
    :param iterator: iterator e.g.   image_raster_iter
    :param kwargs: Parameters for functions in components
    :return: graph
    """
    
    def accumulate_results(results, **kwargs):
        newim = copy_image(im)
        i = 0
        for dpatch in iterator(newim, **kwargs):
            dpatch.data[...] = results[i].data[...]
            i += 1
        return newim
    
    results = [arlexecute.execute(imagefunction(copy_image(dpatch))) for dpatch in iterator(im, **kwargs)]
    
    return arlexecute.execute(accumulate_results, pure=True)(results, **kwargs)


def generic_image_arlexecute(image_unary_function, im: Image, facets=4, overlap=0, **kwargs) :
    """ Definition of interface for generic_image_arlexecute using scatter/gather

    This generates a graph for imagefunction. Note that im cannot be a graph itself.

    :param func: image_unary_function: Function to be applied to all pixels
    :param im: Image to be processed
    :param facets: Number of facets on each axis
    :param overlap: Overlap of facets in pixels
    :param kwargs: Parameters for functions in components
    :return: graph
    """
    output = arlexecute.execute(create_empty_image_like, nout=1, pure=True)(im)
    scattered = arlexecute.execute(image_scatter_facets, pure=True, nout=facets ** 2)(im, facets=facets,
                                                                                      overlap=overlap)
    result = [arlexecute.execute(image_unary_function)(s, **kwargs) for s in scattered]
    return arlexecute.execute(image_gather_facets, nout=1, pure=True)(result, output, facets=facets, overlap=overlap)