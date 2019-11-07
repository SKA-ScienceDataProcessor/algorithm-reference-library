"""Unit tests for testing support


"""
import logging
import sys

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from data_models.polarisation import PolarisationFrame
from processing_components.imaging.base import create_image_from_visibility
from processing_components.imaging.primary_beams import create_pb
from processing_components.simulation.configurations import create_named_configuration
from processing_components.visibility.base import create_visibility
from workflows.arlexecute.image.image_arlexecute import image_arlexecute_map_workflow
from wrappers.arlexecute.execution_support.arlexecute import arlexecute
from wrappers.arlexecute.execution_support.dask_init import get_dask_Client
from wrappers.arlexecute.image.operations import export_image_to_fits

log = logging.getLogger()
log.setLevel(logging.DEBUG)
#log.addHandler(logging.StreamHandler(sys.stdout))
#log.addHandler(logging.StreamHandler(sys.stderr))
import asyncio

logging.getLogger('asyncio').setLevel(logging.WARNING)

if __name__ == '__main__':
    client = get_dask_Client(threads_per_worker=1,
                             processes=True,
                             memory_limit=32 * 1024 * 1024 * 1024,
                             n_workers=8)
    
    arlexecute.set_client(client=client)
    
    from data_models.parameters import arl_path
    
    dir = arl_path('test_results')
    
    frequency = numpy.linspace(1e8, 1.5e8, 3)
    channel_bandwidth = numpy.array([2.5e7, 2.5e7, 2.5e7])
    flux = numpy.array([[100.0], [100.0], [100.0]])
    phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
    config = create_named_configuration('LOWBD2-CORE')
    times = numpy.linspace(-300.0, 300.0, 3) * numpy.pi / 43200.0
    nants = config.xyz.shape[0]
    assert nants > 1
    assert len(config.names) == nants
    assert len(config.mount) == nants
    
    config = create_named_configuration('LOWBD2', rmax=1000.0)
    phasecentre = SkyCoord(ra=+15 * u.deg, dec=-45.0 * u.deg, frame='icrs', equinox='J2000')
    vis = create_visibility(config, times, frequency,
                            channel_bandwidth=channel_bandwidth,
                            phasecentre=phasecentre, weight=1.0,
                            polarisation_frame=PolarisationFrame('stokesI'))
    
    model = create_image_from_visibility(vis, npixel=4096, cellsize=0.001, override_cellsize=False)
    beam = image_arlexecute_map_workflow(model, create_pb, facets=16, pointingcentre=phasecentre,
                                         telescope='MID')
    beam = arlexecute.compute(beam, sync=True)
    from time import sleep
    
    sleep(10)
    exit()
    
    assert numpy.max(beam.data) > 0.0
    export_image_to_fits(beam, "cluster_test_image.fits")
