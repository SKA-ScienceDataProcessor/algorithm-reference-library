"""
Prepare large images for deconvolution tests.

    - Simulates a LOW observation only including stations up to a given radius
    - Avoids wide-field imaging problems by setting w to zero in all steps
    - Uses dask graphs to simulate, predict, and invert
    - Allows different ways of distributing the processing over multiple nodes
    
"""
import os
import sys

sys.path.append(os.path.join('..', '..'))

results_dir = './results'
os.makedirs(results_dir, exist_ok=True)

import numpy

from astropy.coordinates import SkyCoord
from astropy import units as u
from arl.data.polarisation import PolarisationFrame
from arl.image.operations import export_image_to_fits, smooth_image
from arl.imaging import advise_wide_field

from arl.graphs.dask_init import get_dask_Client
from arl.graphs.delayed import compute_list, create_invert_graph, create_predict_graph, create_weight_vis_graph_list
from arl.util.graph_support import create_simulate_vis_graph, create_gleam_model_graph

import logging

log = logging.getLogger()
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))

if __name__ == '__main__':
    
    c = get_dask_Client()
    
    # Create images for these maximum array extents.
    rmax_range = [750.0, 1.5e3, 3e3, 6e3, 1.2e4]
    # Image sizes are 1024, 2048, 4096, 8192, 16384
    
    nfreqwin = 7
    ntimes = 3
    frequency = numpy.linspace(0.8e8, 1.2e8, nfreqwin)
    channel_bandwidth = numpy.array(nfreqwin * [frequency[1] - frequency[0]])

    times = numpy.linspace(-numpy.pi / 3.0, numpy.pi / 3.0, ntimes)
    
    phasecentre = SkyCoord(ra=+30.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')
    order = None # Different vis_graphs for each time: 'frequency' | 'time' | 'both' | None

    for rmax in rmax_range:
        b = 2 * rmax
        cellsize = 0.00001 * (8e4 / b)
        npixel = int(32 * 1024 * rmax / 8e4)
        padding = 2
        
        vis_graph_list = create_simulate_vis_graph('LOWBD2',
                                                   frequency=frequency,
                                                   channel_bandwidth=channel_bandwidth,
                                                   times=times,
                                                   phasecentre=phasecentre,
                                                   rmax=rmax,
                                                   zerow=True,
                                                   format='vis', order=order)
        
        log.info('rmax is %.1f (m)' % (rmax))
        log.info('Observing times %s' % (times))
        log.info("Observing frequencies %s Hz" % (frequency))
        log.info("Number of pixels %d" % (npixel))
        log.info("Cellsize = %.6f radians" % (cellsize))
        
        vis_graph_list = compute_list(c, vis_graph_list)
        
        advice = advise_wide_field(vis_graph_list[0], guard_band_image=4.0, delA=0.02, wprojection_planes=1)
        vis_slices = advice['vis_slices']
        npixel = advice['npixels2']
        cellsize = advice['cellsize']

        future = c.compute(delayed(create_low_test_image_from_gleam)(vis_graph_list[0], npixel=npixel, nchan=1,
                                                    cellsize=cellsize, frequency=[frequency[0]],
                                                    channel_bandwidth=[channel_bandwidth[0]],
                                                    polarisation_frame=PolarisationFrame("stokesI")))
        model = future.result()

        cmodel = smooth_image(model)
        export_image_to_fits(cmodel, '%s/imaging-low-cmodel_npixel_%d.fits' % (results_dir, npixel))
        cmodel = None

        predicted_vis_graph_list = create_predict_graph(vis_graph_list, model)
        predicted_vis_graph_list = create_weight_vis_graph_list(predicted_vis_graph_list, model)
        predicted_vis_graph_list = compute_list(c, predicted_vis_graph_list)
        
        # Make the dirty image and point spread function
        future = c.compute(create_invert_graph(predicted_vis_graph_list, model))
        dirty, sumwt = future.result()
        print("Max, min in dirty image = %.6f, %.6f, sumwt = %s" % (dirty.data.max(), dirty.data.min(), sumwt))
        export_image_to_fits(dirty, '%s/imaging-low-dirty_npixel_%d.fits' % (results_dir, npixel))
        dirty = None
        
        future = c.compute(create_invert_graph(predicted_vis_graph_list, model, dopsf=True))
        psf, sumwt = future.result()
        print("Max, min in PSF         = %.6f, %.6f, sumwt = %s" % (psf.data.max(), psf.data.min(), sumwt))
        
        export_image_to_fits(psf, '%s/imaging-low-psf_npixel_%d.fits' % (results_dir, npixel))
    
    exit()
