""" Generic functions converted to Dask.delayed graphs. `Dask <http://dask.pydata.org/>`_ is a python-based
flexible parallel computing library for analytic computing. Dask.delayed can be used to wrap functions for deferred
execution
thus allowing construction of graphs.

For example, consider a trivial example to take the square root of an image::

        def imagerooter(im, **kwargs):
            im.data = numpy.sqrt(numpy.abs(im.data))
            return im
        root = create_generic_image_graph(imagerooter, myimage,  image_raster_iter, facets=4).compute()

We create the graph and execute it immediately. We are using the  image_raster_iter so the image will be divided into 16
subimages and passed to processing by imagerooter, and then the answers are reassembled.

We  could keep the graph and use it in other graphs. See the imaging-dask note book for more detail.
"""

from dask import delayed

from arl.data.data_models import Image
from arl.image.operations import copy_image, create_empty_image_like
from arl.image.gather_scatter import image_gather_facets, image_scatter_facets


def create_generic_blockvisibility_graph(visfunction, vis_graph_list, additive=True, *args,
                                         **kwargs):
    """ Definition of interface for create_generic_blockvisibility_graph_visfunction.

    :func visfunction: Function to be applied
    :param vis_graph_list: List of vis_graphs
    :param additive: Add to existing visibility? (True)
    :param args:
    :param kwargs: Parameters for functions in graphs
    :return: List of graphs
    """
    
    def accumulate_results(results, **kwargs):
        for i, result in enumerate(results):
            if additive:
                vis_graph_list[i].data['vis'] += result.data['vis']
            else:
                vis_graph_list[i].data['vis'] = result.data['vis']
        return vis_graph_list
    
    results = list()
    for vis_graph in vis_graph_list:
        results.append(delayed(visfunction, pure=True)(vis_graph, *args, **kwargs))
    return [delayed(accumulate_results, pure=True)(results, **kwargs)]


def create_generic_image_iterator_graph(imagefunction, im: Image, iterator, **kwargs) -> delayed:
    """ Definition of interface for create_generic_image_graph
    
    This generates a graph for imagefunction. Note that im cannot be a graph itself.

    :func imagefunction: Function to be applied to all pixels
    :param im: Image to be processed
    :param iterator: iterator e.g.   image_raster_iter
    :param kwargs: Parameters for functions in graphs
    :return: graph
    """
    
    def accumulate_results(results, **kwargs):
        newim = copy_image(im)
        i = 0
        for dpatch in iterator(newim, **kwargs):
            dpatch.data[...] = results[i].data[...]
            i += 1
        return newim
    
    results = list()
    
    for dpatch in iterator(im, **kwargs):
        results.append(delayed(imagefunction(copy_image(dpatch), **kwargs)))
    
    return delayed(accumulate_results, pure=True)(results, **kwargs)


def create_generic_image_graph(image_unary_function, im: Image, facets=4, **kwargs) -> delayed:
    """ Definition of interface for create_generic_image_graph using scatter/gather

    This generates a graph for imagefunction. Note that im cannot be a graph itself.

    :func image_unary_function: Function to be applied to all pixels
    :param im: Image to be processed
    :param facets: Number of facets on each axis
    :param kwargs: Parameters for functions in graphs
    :return: graph
    """
    output = delayed(create_empty_image_like, nout=1, pure=True)(im)
    scattered = delayed(image_scatter_facets, pure=True, nout=facets ** 2)(im, facets=facets)
    result = [delayed(image_unary_function)(s) for s in scattered]
    return delayed(image_gather_facets, nout=1, pure=True)(result, output, facets=facets)
