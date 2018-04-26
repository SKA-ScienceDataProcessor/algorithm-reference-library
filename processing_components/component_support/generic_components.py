""" Generic functions converted to arlexecute components. arlexecute is a wrapper for`Dask <http://dask.pydata.org/>`_
is a python-based flexible parallel computing library for analytic computing. Dask.delayed can be used to wrap
functions for deferred execution thus allowing construction of components.

For example, consider a trivial example to take the square root of an image::

        def imagerooter(im, **kwargs):
            im.data_models = numpy.sqrt(numpy.abs(im.data_models))
            return im
        root = generic_image_component(imagerooter, myimage,  image_raster_iter, facets=4).compute()

We create the graph and execute it immediately. We are using the  image_raster_iter so the image will be divided into 16
subimages and passed to processing by imagerooter, and then the answers are reassembled.

We  could keep the graph and use it in other components. See the imaging-dask note book for more detail.
"""

from processing_components.component_support.arlexecute import arlexecute

from data_models.memory_data_models import Image
from libs.image.operations import copy_image, create_empty_image_like
from libs.image.gather_scatter import image_gather_facets, image_scatter_facets


def generic_blockvisibility_component(visfunction, vis_list, additive=True, *args, **kwargs):
    """ Definition of interface for create_generic_blockvisibility_graph_visfunction.

    :func visfunction: Function to be applied
    :param vis_list: List of vis_graphs
    :param additive: Add to existing visibility? (True)
    :param args:
    :param kwargs: Parameters for functions in components
    :return: List of components
    """
    
    def accumulate_results(results, **kwargs):
        for i, result in enumerate(results):
            if additive:
                vis_list[i].data['vis'] += result.data['vis']
            else:
                vis_list[i].data['vis'] = result.data['vis']
        return vis_list
    
    results = list()
    for vis_graph in vis_list:
        results.append(arlexecute.execute(visfunction, pure=True)(vis_graph, *args, **kwargs))
    return [arlexecute.execute(accumulate_results, pure=True)(results, **kwargs)]


def generic_image_iterator_component(imagefunction, im: Image, iterator, **kwargs):
    """ Definition of interface for generic_image_component
    
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


def generic_image_component(image_unary_function, im: Image, facets=4, **kwargs) :
    """ Definition of interface for generic_image_component using scatter/gather

    This generates a graph for imagefunction. Note that im cannot be a graph itself.

    :func image_unary_function: Function to be applied to all pixels
    :param im: Image to be processed
    :param facets: Number of facets on each axis
    :param kwargs: Parameters for functions in components
    :return: graph
    """
    output = arlexecute.execute(create_empty_image_like, nout=1, pure=True)(im)
    scattered = arlexecute.execute(image_scatter_facets, pure=True, nout=facets ** 2)(im, facets=facets)
    result = [arlexecute.execute(image_unary_function)(s) for s in scattered]
    return arlexecute.execute(image_gather_facets, nout=1, pure=True)(result, output, facets=facets)
