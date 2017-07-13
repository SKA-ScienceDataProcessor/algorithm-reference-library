""" Visibility iterators for iterating through a BlockVisibility or Visibility.

A typical use would be to make a sequence of snapshot visibilitys::

    for rows in vis_timeslice_iter(vt):
        visslice = create_visibility_from_rows(vt, rows)
        dirtySnapshot = create_visibility_from_visibility(visslice, npixel=512, cellsize=0.001, npol=1)
        dirtySnapshot, sumwt = invert_2d(visslice, dirtySnapshot)


"""

import logging

import numpy

from arl.visibility.operations import create_visibility_from_rows
from arl.visibility.coalesce import coalesce_visibility, decoalesce_visibility
from arl.data.data_models import Visibility, BlockVisibility
from arl.visibility.iterators import vis_slice_iter, vis_timeslice_iter, vis_wstack_iter

log = logging.getLogger(__name__)


def visibility_scatter(vis, vis_iter, **kwargs):
    """Scatter an visibility into a list of subvisibilities

    :param vis: Visibility
    :param vis_iter: visibility iterator
    :returns: list of subvisibilitys
    """
    
    visibility_list = list()
    for i, rows in enumerate(vis_iter(vis, **kwargs)):
        if rows is not None:
            subvis = create_visibility_from_rows(vis, rows)
            visibility_list.append(subvis)
        else:
            visibility_list.append(None)
    
    return visibility_list

def visibility_gather(visibility_list, vis, vis_iter, **kwargs):
    """Gather a list of subvisibilities back into an visibility

    :param visibility_list: List of subvisibilities
    :param vis: Output visibility
    :param vis_iter: visibility iterator
    :returns: vis
    """
    
    for i, rows in enumerate(vis_iter(vis, **kwargs)):
        assert i < len(visibility_list), "Gather not consistent with scatter"
        if rows is not None and visibility_list[i] is not None:
            vis.data[rows] = visibility_list[i].data[...]
    
    return vis

def visibility_scatter_index(vis, **kwargs):
    return visibility_scatter(vis, vis_iter=vis_slice_iter, **kwargs)

def visibility_scatter_w(vis: Visibility, **kwargs):
    if type(vis) == BlockVisibility:
        avis = coalesce_visibility(vis, **(kwargs))
        return visibility_scatter(avis, vis_iter=vis_wstack_iter, **kwargs)
    else:
        return visibility_scatter(vis, vis_iter=vis_wstack_iter, **kwargs)

def visibility_scatter_time(vis, **kwargs):
    return visibility_scatter(vis, vis_iter=vis_timeslice_iter, **kwargs)

def visibility_gather_index(visibility_list, vis, **kwargs):
    return visibility_gather(visibility_list, vis, vis_iter=vis_slice_iter, **kwargs)

def visibility_gather_w(visibility_list, vis: Visibility, **kwargs):
    if type(vis) == BlockVisibility:
        cvis = coalesce_visibility(vis, **kwargs)
        return decoalesce_visibility(visibility_gather(visibility_list, cvis, vis_iter=vis_wstack_iter,
                                                       **kwargs))
    else:
        return visibility_gather(visibility_list, vis, vis_iter=vis_wstack_iter, **kwargs)

def visibility_gather_time(visibility_list, vis, **kwargs):
    return visibility_gather(visibility_list, vis, vis_iter=vis_timeslice_iter, **kwargs)