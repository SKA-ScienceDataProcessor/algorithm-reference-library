""" Generic functions converted to Dask.delayed graphs. `Dask <http://dask.pydata.org/>`_ is a python-based
flexible parallel computing library for analytic computing. Dask.delayed can be used to wrap functions for deferred
execution
thus allowing construction of graphs.

For example, consider a trivial example to take the square root of an image::

        def imagerooter(im, **kwargs):
            im.data = numpy.sqrt(numpy.abs(im.data))
            return im
        root = create_generic_image_graph(imagerooter)(myimage, raster_iter, facets=4).compute()

We create the graph and execute it immediately. We are using the raster_iter so the image will be divided into 16
subimages and passed to processing by imagerooter, and then the answers are reassembled.

We  could keep the graph and use it in  other graphs. See the imaging-dask note book for more detail.
"""

import collections

from dask import delayed

from arl.data.data_models import BlockVisibility, GainTable, Image
from arl.image.operations import copy_image
from arl.visibility.operations import create_visibility_from_rows, \
    copy_visibility, create_blockvisibility_from_rows


def create_generic_blockvisibility_graph(visfunction):
    
    """ Wrap a generic function into a graph

    This returns a graph for a generic visibility function distributed via the iterator.
    
    :param visfunction: Visibility function
    :returns: function to generate a graph
   """
    
    def create_generic_blockvisibility_graph_visfunction(vis: BlockVisibility,
                                                         iterator, *args, **kwargs):
        """ Definition of interface for create_generic_blockvisibility_graph_visfunction.
        
        Note that vis cannot be a graph.
        
        :param vis:
        :param iterator:
        :param args:
        :param kwargs:
        :return:
        """
    
        def accumulate_results(results, **kwargs):
            i = 0
            for rows in iterator(vis, **kwargs):
                vis.data['vis'][rows] += results[i].data['vis']
                i += 1
            return vis

        results = list()

        for rows in iterator(vis, **kwargs):
            visslice = copy_visibility(create_blockvisibility_from_rows(vis, rows))
            results.append(delayed(visfunction, pure=True)(visslice, *args, **kwargs))
        return delayed(accumulate_results, pure=True)(results, **kwargs)

    return create_generic_blockvisibility_graph_visfunction

def create_generic_image_graph(imagefunction):
    """ Wrap an image function into a function that returns a graph
    
    This works just as Dask.delayed. It returns a function that can be called
    to return a graph.
    
    An iterator, e.g. raster_iter, is used to iterate over the image. The image function is
    called and the result inserted into the image.
    
    :param imagefunction: Function to be turned into a graph
    :return: function to create a graph
    
    """

    def create_generic_image_graph_imagefunction(im: Image, iterator, **kwargs):
        """ Definition of interface for create_generic_image_graph
        
        This generates a graph for imagefunction. Note that im cannot be a graph itself.

        :param im: Image to be processed
        :param iterator: iterator e.g. raster_iter
        :param kwargs:
        :return: graph
        """
    
        def accumulate_results(results, **kwargs):
            newim = copy_image(im)
            i=0
            for dpatch in iterator(newim, **kwargs):
                dpatch.data[...] = results[i].data[...]
                i+=1
            return newim

        results = list()

        for dpatch in iterator(im, **kwargs):
            results.append(delayed(imagefunction(copy_image(dpatch), **kwargs)))
            
        return delayed(accumulate_results, pure=True)(results, **kwargs)

    return create_generic_image_graph_imagefunction
