""" Pipelines expressed as dask graphs
"""

from astropy import units as u
from astropy.coordinates import SkyCoord
from dask import delayed

from arl.calibration.operations import apply_gaintable, create_gaintable_from_blockvisibility
from arl.data.persist import arl_dump, arl_load
from arl.data.polarisation import PolarisationFrame
from arl.graphs.graphs import create_predict_wstack_graph
from arl.util.testing_support import create_named_configuration, simulate_gaintable, \
    create_low_test_image_from_gleam, create_low_test_beam
from arl.visibility.base import create_blockvisibility, create_visibility

import logging

log = logging.getLogger(__name__)

def create_simulate_vis_graph(config='LOWBD2-CORE',
                              phasecentre=SkyCoord(ra=+15.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000'),
                              frequency=None, channel_bandwidth=None, times=None,
                              polarisation_frame=PolarisationFrame("stokesI"), order='frequency',
                              format='blockvis',
                              **kwargs) -> delayed:
    """ Create a graph to simulate an observation
    
    The simulation step can generate a single BlockVisibility or a list of BlockVisibility's.
    The parameter keyword determines the way that the list is constructed.
    If order='frequency' then len(frequency) BlockVisibility's with all times are created.
    If order='time' then  len(times) BlockVisibility's with all frequencies are created.
    If order = 'both' then len(times) * len(times) BlockVisibility's are created each with
    a single time and frequency. If order = None then all data are created in one BlockVisibility.
    
    The output format can be either 'blockvis' (for calibration) or 'vis' (for imaging)
    
    :param config: Name of configuration: def LOWBDS-CORE
    :param phasecentre: Phase centre def: SkyCoord(ra=+15.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')
    :param frequency: def [1e8]
    :param channel_bandwidth: def [1e6]
    :param times: Observing times in radians: def [0.0]
    :param polarisation_frame: def PolarisationFrame("stokesI")
    :param order: 'time' or 'frequency' or 'both' or None: def 'frequency'
    :param format: 'blockvis' or 'vis': def 'blockvis'
    :param kwargs:
    :return: vis_graph_list with different frequencies in different elements
    """
    if format == 'vis':
        create_vis = create_visibility
    else:
        create_vis = create_blockvisibility
        
    if times is None:
        times = [0.0]
    if channel_bandwidth is None:
        channel_bandwidth = [1e6]
    if frequency is None:
        frequency = [1e8]
    conf = create_named_configuration(config, **kwargs)
    
    if order =='time':
        log.debug("create_simulate_vis_graph: Simulating distribution in %s" % order)
        vis_graph_list = list()
        for i, time in enumerate(times):
            vis_graph_list.append(delayed(create_vis, nout=1)(conf, [times[i]], frequency=frequency,
                                                                channel_bandwidth=channel_bandwidth,
                                                                weight=1.0, phasecentre=phasecentre,
                                                                polarisation_frame=polarisation_frame, **kwargs))
            
    elif order == 'frequency':
        log.debug("create_simulate_vis_graph: Simulating distribution in %s" % order)
        vis_graph_list = list()
        for j, _ in enumerate(frequency):
            vis_graph_list.append(delayed(create_vis, nout=1)(conf, times, frequency=[frequency[j]],
                                                                channel_bandwidth=[channel_bandwidth[j]],
                                                                weight=1.0, phasecentre=phasecentre,
                                                                polarisation_frame=polarisation_frame, **kwargs))

    elif order =='both':
        log.debug("create_simulate_vis_graph: Simulating distribution in time and frequency")
        vis_graph_list = list()
        for i, _ in enumerate(times):
            for j, _ in enumerate(frequency):
                vis_graph_list.append(delayed(create_vis, nout=1)(conf, [times[i]], frequency=[frequency[j]],
                                                                    channel_bandwidth=[channel_bandwidth[j]],
                                                                    weight=1.0, phasecentre=phasecentre,
                                                                    polarisation_frame=polarisation_frame, **kwargs))

    elif order is None:
        log.debug("create_simulate_vis_graph: Simulating into single %s" % format)
        vis_graph_list = list()
        vis_graph_list.append(delayed(create_vis, nout=1)(conf, times, frequency=frequency,
                                                      channel_bandwidth=channel_bandwidth,
                                                      weight=1.0, phasecentre=phasecentre,
                                                      polarisation_frame=polarisation_frame, **kwargs))
    else:
        raise NotImplemented("order $s not known" % order)
    return vis_graph_list


def create_predict_gleam_model_graph(vis_graph_list, frequency, channel_bandwidth,
                                     npixel=512, cellsize=0.001,
                                     c_predict_graph=create_predict_wstack_graph, **kwargs):
    """ Create a graph to fill in a model with the gleam sources and predict into a vis_graph_list

    :param vis_graph_list:
    :param npixel: 512
    :param cellsize: 0.001
    :param c_predict_graph: def create_predict_wstack_graph
    :param kwargs:
    :return: vis_graph_list
    """
    
    # Note that each vis_graph has it's own model_graph
    
    predicted_vis_graph_list = list()
    for i, vis_graph in enumerate(vis_graph_list):
        model_graph = create_gleam_model_graph(vis_graph, frequency, channel_bandwidth, npixel=npixel,
                                               cellsize=cellsize, **kwargs)
        predicted_vis_graph_list.append(c_predict_graph([vis_graph], model_graph, **kwargs)[0])
    return predicted_vis_graph_list


def create_gleam_model_graph(vis_graph: delayed, frequency, channel_bandwidth, npixel=512, cellsize=0.001,
                             facets=4, **kwargs):
    """ Create a graph to fill in a model with the gleam sources
    
    This spreads the work over facet**2 nodes

    :param vis_graph: Single vis_graph
    :param frequency:
    :param channel_bandwidth:
    :param npixel: 512
    :param cellsize: 0.001
    :param facets: def 4
    :param kwargs:
    :return: graph
    """
    
    def calculate_gleam_model(vis):
        model = create_low_test_image_from_gleam(npixel=npixel, frequency=frequency,
                                                 channel_bandwidth=channel_bandwidth,
                                                 cellsize=cellsize, phasecentre=vis.phasecentre)
        beam = create_low_test_beam(model)
        model.data *= beam.data
        return model
        
    return delayed(calculate_gleam_model, nout=1)(vis_graph)
    
def create_corrupt_vis_graph(vis_graph_list, gt_graph=None, **kwargs):
    """ Create a graph to apply gain errors to a vis_graph_list
    
    :param vis_graph_list:
    :param gt_graph: Optional gain table graph
    :param kwargs:
    :return:
    """
    
    def corrupt_vis(vis, gt, **kwargs):
        if gt is None:
            gt = create_gaintable_from_blockvisibility(vis, **kwargs)
            gt = simulate_gaintable(gt, **kwargs)
        return apply_gaintable(vis, gt)

    return [delayed(corrupt_vis, nout=1)(vis_graph, gt_graph, **kwargs) for vis_graph in vis_graph_list]


def create_dump_vis_graph(vis_graph_list, name='imaging_dask') -> delayed:
    """ Create a graph to save a vis_graph_list
    
    :param vis_graph_list:
    :param name:
    :return:
    """
    def save_all(vis):
        for i, v in enumerate(vis):
            arl_dump(v, "%s_%d.pickle" % (name, i))
            
    return delayed(save_all)(vis_graph_list)

def create_load_vis_graph(name='imaging_dask'):
    """ Load's pickled data with the glob of "%s_*.pickle" % (name)
    
    :param name: Root of name
    :return: vis_graph_list
    """
    import glob
    name = "%s_*.pickle" % (name)
    persist_list = glob.glob(name)
    assert len(persist_list)
    return [delayed(arl_load, nout=1)(name) for name in persist_list]
