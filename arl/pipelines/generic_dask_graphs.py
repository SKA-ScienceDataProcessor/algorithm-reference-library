import collections

from dask import delayed

from arl.calibration.solvers import solve_gaintable, apply_gaintable
from arl.calibration.operations import append_gaintable
from arl.data.data_models import BlockVisibility, GainTable, Image
from arl.data.parameters import get_parameter
from arl.fourier_transforms.ftprocessor import invert_2d, residual_image, \
    predict_skycomponent_blockvisibility, \
    invert_timeslice_single, predict_timeslice_single, normalize_sumwt
from arl.image.deconvolution import deconvolve_cube, restore_cube
from arl.image.operations import copy_image
from arl.visibility.iterators import vis_timeslice_iter
from arl.visibility.operations import create_visibility_from_rows, \
    copy_visibility, create_blockvisibility_from_rows


def create_generic_blockvisibility_graph(visfunction):
    
    """ Wrap a generic function into a graph

    This returns a graph for a generic visibility function distributed via the iterator.
    
    :param vis: Visibility or Union(Visibility, Iterable)
    :param visfunction: Visibility function
    :param iterator: visibility iterator returning rows
    :param *args: Arguments for visfunction
    :param kwargs: Keyword values for visfunction
    :returns: gaintable
   """
    
    def create_generic_blockvisibility_graph_visfunction(vis: BlockVisibility,
                                                         iterator, *args, **kwargs):
        """ Definition of interface for create_generic_blockvisibility_graph_visfunction
        
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

    :param imagefunction: Function to be turned into a graph
    :return: function to create a graph
    
    :param im:
    :param iterator:
    :param args:
    :param kwargs:
    :return:
    """

    def create_generic_image_graph_imagefunction(im: Image, iterator, **kwargs):
        """ Definition of interface for create_generic_image_graph

        :param im:
        :param iterator:
        :param args:
        :param kwargs:
        :return:
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
