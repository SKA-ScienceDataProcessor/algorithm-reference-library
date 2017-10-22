""" Visibility iterators for iterating through a BlockVisibility or Visibility.

A typical use would be to make a sequence of snapshot visibilitys::

    for rows in vis_timeslice_iter(vt):
        visslice = create_visibility_from_rows(vt, rows)
        dirtySnapshot = create_visibility_from_visibility(visslice, npixel=512, cellsize=0.001, npol=1)
        dirtySnapshot, sumwt = invert_2d(visslice, dirtySnapshot)


"""

import logging
from typing import List

import numpy

from arl.data.data_models import Visibility, BlockVisibility
from arl.visibility.coalesce import coalesce_visibility, decoalesce_visibility
from arl.visibility.iterators import vis_slice_iter, vis_timeslice_iter, vis_wstack_iter
from arl.visibility.base import create_visibility_from_rows

log = logging.getLogger(__name__)


def visibility_scatter(vis: Visibility, vis_iter, vis_slices=1, **kwargs) -> List[Visibility]:
    """Scatter a visibility into a list of subvisibilities

    :param vis: Visibility
    :param vis_iter: visibility iterator
    :return: list of subvisibilitys
    """
    
    visibility_list = list()
    for i, rows in enumerate(vis_iter(vis, vis_slices=vis_slices, **kwargs)):
        subvis = create_visibility_from_rows(vis, rows)
        visibility_list.append(subvis)
        
    return visibility_list


def visibility_gather(visibility_list: List[Visibility], vis: Visibility, vis_iter, vis_slices=1,
                      **kwargs) -> Visibility:
    """Gather a list of subvisibilities back into a visibility
    
    The iterator setup must be the same as used in the scatter.

    :param visibility_list: List of subvisibilities
    :param vis: Output visibility
    :param vis_iter: visibility iterator
    :return: vis
    """
    
    for i, rows in enumerate(vis_iter(vis, vis_slices=vis_slices, **kwargs)):
        assert i < len(visibility_list), "Gather not consistent with scatter"
        if numpy.sum(rows) and visibility_list[i] is not None:
            vis.data[rows] = visibility_list[i].data[...]
    
    return vis


def visibility_scatter_index(vis: Visibility, **kwargs) -> List[Visibility]:
    return visibility_scatter(vis, vis_iter=vis_slice_iter, **kwargs)


def visibility_scatter_w(vis: Visibility, **kwargs) -> List[Visibility]:
    if isinstance(vis, BlockVisibility):
        avis = coalesce_visibility(vis, **(kwargs))
        visibility_list = visibility_scatter(avis, vis_iter=vis_wstack_iter, **kwargs)
    else:
        visibility_list = visibility_scatter(vis, vis_iter=vis_wstack_iter, **kwargs)
        
    return visibility_list


def visibility_scatter_time(vis: Visibility, **kwargs) -> List[Visibility]:
    return visibility_scatter(vis, vis_iter=vis_timeslice_iter, **kwargs)


def visibility_gather_index(visibility_list: List[Visibility], vis: Visibility, **kwargs) -> Visibility:
    return visibility_gather(visibility_list, vis, vis_iter=vis_slice_iter, **kwargs)


def visibility_gather_w(visibility_list: List[Visibility], vis: Visibility, **kwargs) -> Visibility:
    if isinstance(vis, BlockVisibility):
        cvis = coalesce_visibility(vis, **kwargs)
        return decoalesce_visibility(visibility_gather(visibility_list, cvis, vis_iter=vis_wstack_iter, **kwargs))
    else:
        return visibility_gather(visibility_list, vis, vis_iter=vis_wstack_iter, **kwargs)


def visibility_gather_time(visibility_list: List[Visibility], vis: Visibility, **kwargs) -> Visibility:
    return visibility_gather(visibility_list, vis, vis_iter=vis_timeslice_iter, **kwargs)


def visibility_scatter_channel(vis: BlockVisibility, **kwargs) -> List[Visibility]:
    """ Scatter in frequency
    
    :param vis:
    :param kwargs:
    :return:
    """
    def extract_channel(v, chan):
        vis_shape = numpy.array(v.data['vis'].shape)
        vis_shape[3] = 1

        vis = BlockVisibility(data=None,
                              frequency=numpy.array([v.frequency[chan]]),
                              channel_bandwidth=numpy.array([v.channel_bandwidth[chan]]),
                              phasecentre=v.phasecentre,
                              configuration=v.configuration,
                              uvw=v.uvw,
                              time=v.time,
                              vis=v.vis[..., chan, :][..., numpy.newaxis, :],
                              weight=v.weight[..., chan, :][..., numpy.newaxis, :],
                              integration_time=v.integration_time,
                              polarisation_frame=v.polarisation_frame)
        return vis
    
    return [extract_channel(vis, channel) for channel, _ in enumerate(vis.frequency)]


def visibility_gather_channel(vis_list: List[Visibility], vis: Visibility = None, **kwargs):
    """ Gather a visibility by channel
    
    :param vis_list:
    :param vis:
    :param kwargs:
    :return:
    """
    
    cols = ['vis', 'weight']
    
    if vis is None:

        vis_shape = numpy.array(vis_list[0].vis.shape)
        vis_shape[-2] = len(vis_list)
        for v in vis_list:
            assert len(v.frequency) == 1
            assert len(v.channel_bandwidth) == 1
        vis = BlockVisibility(data=None,
                              frequency=numpy.array([v.frequency[0] for v in vis_list]),
                              channel_bandwidth=numpy.array([v.channel_bandwidth[0] for v in vis_list]),
                              phasecentre=vis_list[0].phasecentre,
                              configuration=vis_list[0].configuration,
                              uvw=vis_list[0].uvw,
                              time=vis_list[0].time,
                              vis=numpy.zeros(vis_shape, dtype=vis_list[0].vis.dtype),
                              weight=numpy.ones(vis_shape, dtype=vis_list[0].weight.dtype),
                              integration_time=vis_list[0].integration_time,
                              polarisation_frame=vis_list[0].polarisation_frame)
    
    assert len(vis.frequency) == len(vis_list)
    
    for chan, _ in enumerate(vis_list):
        subvis = vis_list[chan]
        assert abs(subvis.frequency[0] - vis.frequency[chan]) < 1e-15
        for col in cols:
            vis.data[col][..., chan, :] = subvis.data[col][..., 0, :]
        vis.frequency[chan] = subvis.frequency[0]
        
    nchan = vis.vis.shape[-2]
    assert nchan == len(vis.frequency)
    
    return vis
