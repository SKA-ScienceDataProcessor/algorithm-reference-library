""" Pipelines expressed as dask graphs
"""

from typing import List, Union
from astropy import units as u
from astropy.coordinates import SkyCoord
from dask import delayed

from arl.calibration.operations import apply_gaintable, create_gaintable_from_blockvisibility
from arl.data.persist import arl_dump, arl_load
from arl.data.polarisation import PolarisationFrame
from arl.graphs.dask_graphs import create_predict_wstack_graph
from arl.graphs.generic_dask_graphs import create_generic_image_graph
from arl.util.testing_support import create_named_configuration, simulate_gaintable, \
    create_low_test_image_from_gleam, create_low_test_beam
from arl.visibility.operations import create_blockvisibility


def create_simulate_vis_graph(config='LOWBD2-CORE',
                              phasecentre=SkyCoord(ra=+15.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000'),
                              frequency=None, channel_bandwidth=None, times=None,
                              polarisation_frame=PolarisationFrame("stokesI"), order='frequency',
                              **kwargs) -> delayed:
    """ Create a graph to simulate an observation
    
    :param config: Name of configuration: def LOWBDS-CORE
    :param phasecentre: Phase centre as SkyCoord def: SkyCoord(ra=+15.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')
    :param frequency: def [1e8]
    :param channel_bandwidth: def [1e6]
    :param times: Observing times in radians: def [0.0]
    :param polarisation_frame: def PolarisationFrame("stokesI")
    :param order: 'time'|'frequency'|'both': def 'frequency'
    :param kwargs:
    :return: vis_graph_list with different frequencies in different elements
    """

    if times is None:
        times = [0.0]
    if channel_bandwidth is None:
        channel_bandwidth = [1e6]
    if frequency is None:
        frequency = [1e8]
    conf = create_named_configuration(config)
    if order =='time':
        vis_graph_list = list()
        for i, time in enumerate(times):
            vis_graph_list.append(delayed(create_blockvisibility, nout=1)(conf, [time], frequency=frequency,
                                                                channel_bandwidth=channel_bandwidth[i],
                                                                weight=1.0, phasecentre=phasecentre,
                                                                polarisation_frame=polarisation_frame, **kwargs))
    elif order =='both':
        vis_graph_list = list()
        for i, time in enumerate(times):
            for j, freq in enumerate(frequency):
                vis_graph_list.append(delayed(create_blockvisibility, nout=1)(conf, [time[i]], frequency=[frequency[j]],
                                                                    channel_bandwidth=[channel_bandwidth[j]],
                                                                    weight=1.0, phasecentre=phasecentre,
                                                                    polarisation_frame=polarisation_frame, **kwargs))


    else:
        vis_graph_list = list()
        for i, freq in enumerate(frequency):
            vis_graph_list.append(delayed(create_blockvisibility, nout=1)(conf, times, frequency=[freq],
                                                                channel_bandwidth=[channel_bandwidth[i]],
                                                                weight=1.0, phasecentre=phasecentre,
                                                                polarisation_frame=polarisation_frame, **kwargs))

    return vis_graph_list


def create_predict_gleam_model_graph(vis_graph_list: List[delayed], npixel=512, cellsize=0.001,
                                     c_predict_graph=create_predict_wstack_graph, **kwargs) -> List[delayed]:
    """ Create a graph to fill in a model with the gleam sources and predict into a vis_graph_list

    :param vis_graph_list:
    :param npixel: 512
    :param cellsize: 0.001
    :param c_predict_graph: def create_predict_wstack_graph
    :param kwargs:
    :return: vis_graph_list
    """
    
    def flatten_list(l):
        return [item for sublist in l for item in sublist]
    
    # Note that each vis_graph has it's own model_graph
    
    predicted_vis_graph_list = list()
    for i, vis_graph in enumerate(vis_graph_list):
        model_graph = create_gleam_model_graph(vis_graph, npixel=npixel, cellsize=cellsize, **kwargs)
        predicted_vis_graph_list.append(c_predict_graph([vis_graph], model_graph, **kwargs)[0])
    return predicted_vis_graph_list


def create_gleam_model_graph(vis_graph: delayed, npixel=512, cellsize=0.001, facets=4, **kwargs) -> delayed:
    """ Create a graph to fill in a model with the gleam sources
    
    This spreads the work over facet**2 nodes

    :param vis_graph: Single vis_graph
    :param npixel: 512
    :param cellsize: 0.001
    :param facets: def 4
    :param kwargs:
    :return: graph
    """
    
    def calculate_model(vis):
        model = create_low_test_image_from_gleam(npixel=npixel, frequency=vis.frequency,
                                                 channel_bandwidth=vis.channel_bandwidth,
                                                 cellsize=cellsize, phasecentre=vis.phasecentre)
        beam = create_low_test_beam(model)
        model.data *= beam.data
        return model
        
    return delayed(calculate_model, nout=1)(vis_graph)
    
def create_corrupt_vis_graph(vis_graph_list: List[delayed], gt_graph=None, **kwargs) -> List[delayed]:
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


def create_dump_vis_graph(vis_graph_list: List[delayed], name='imaging_dask') -> delayed:
    """ Create a graph to save a vis_graph_list
    
    :param vis_graph_list:
    :param name:
    :return:
    """
    def save_all(vis):
        for i, v in enumerate(vis):
            arl_dump(v, "%s_%d.pickle" % (name, i))
            
    return delayed(save_all)(vis_graph_list)

def create_load_vis_graph(name='imaging_dask') -> List[delayed]:
    """ Load's pickled data with the glob of "%s_*.pickle" % (name)
    
    :param name: Root of name
    :return: vis_graph_list
    """
    import glob
    name = "%s_*.pickle" % (name)
    persist_list = glob.glob(name)
    assert len(persist_list)
    return [delayed(arl_load, nout=1)(name) for name in persist_list]

    