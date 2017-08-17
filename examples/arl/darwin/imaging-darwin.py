
import os
import sys

results_dir = './results'
os.makedirs(results_dir, exist_ok=True)

from dask import delayed

sys.path.insert(1, os.environ['ARL'])

results_dir = './results'
os.makedirs(results_dir, exist_ok=True)

import numpy

from astropy.coordinates import SkyCoord
from astropy import units as u

from arl.data.polarisation import PolarisationFrame
from arl.image.operations import export_image_to_fits, qa_image
from arl.imaging import create_image_from_visibility, advise_wide_field

from arl.graphs.dask_init import get_dask_Client, kill_dask_Client

from arl.graphs.graphs import create_invert_wstack_graph, create_deconvolve_facet_graph, \
    create_residual_wstack_graph, compute_list
from arl.util.graph_support import create_simulate_vis_graph, create_predict_gleam_model_graph 
from arl.pipelines.graphs import create_ical_pipeline_graph

import logging

log = logging.getLogger()
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))

# Make a Dask client to manage the processing. Diagnostics are available at the URL given. Try the Status entry.

c=get_dask_Client()

# We create a graph to make the visibility 

nfreqwin=15
ntimes=3
frequency=numpy.linspace(0.8e8,1.2e8,nfreqwin)
channel_bandwidth=numpy.array(nfreqwin*[frequency[1]-frequency[0]])
times = numpy.linspace(-numpy.pi/3.0, numpy.pi/3.0, ntimes)
phasecentre=SkyCoord(ra=+30.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')

vis_graph_list=create_simulate_vis_graph('LOWBD2-CORE',
                                         frequency=frequency, 
                                         channel_bandwidth=channel_bandwidth,
                                         times=times,
                                         phasecentre=phasecentre)


# Now make a graph to fill with a model drawn from GLEAM 

wprojection_planes=1
advice=advise_wide_field(vis_graph_list[0].compute(), guard_band_image=4.0, delA=0.02,
                         wprojection_planes=wprojection_planes)
vis_slices = advice['vis_slices']


predicted_vis_graph_list = create_predict_gleam_model_graph(vis_graph_list,
                                                            vis_slices=advice['vis_slices'])
predicted_vis_graph_list = compute_list(c, predicted_vis_graph_list)

corrupted_vis_graph_list = create_predict_gleam_model_graph(vis_graph_list, 
                                                            vis_slices=advice['vis_slices'],
                                                            phase_error=1.0)
corrupted_vis_graph_list = compute_list(c, corrupted_vis_graph_list)

# Get the LSM. This is currently blank.

def get_LSM(vt, npixel = 512, cellsize=0.001, reffrequency=[1e8]):
    model = create_image_from_visibility(vt, npixel=npixel, cellsize=cellsize, 
                                         npol=1, frequency=reffrequency,
                                         polarisation_frame=PolarisationFrame("stokesI"))
    return model

model_graph=delayed(get_LSM)(vis_graph_list[len(vis_graph_list)//2])


ical_graph = create_ical_pipeline_graph(corrupted_vis_graph_list, 
                                        model_graph=model_graph,  
                                        c_deconvolve_graph=create_deconvolve_facet_graph,
                                        c_invert_graph=create_invert_wstack_graph,
                                        c_residual_graph=create_residual_wstack_graph,
                                        vis_slices=vis_slices, 
                                        algorithm='hogbom', niter=1000, 
                                        fractional_threshold=0.1,
                                        threshold=0.1, nmajor=5, 
                                        gain=0.1, first_selfcal=1,
                                        global_solution=True)


future=c.compute(ical_graph)
deconvolved = future.result()[0]
residual = future.result()[1]
restored = future.result()[2]

print(qa_image(deconvolved, context='Clean image'))

print(qa_image(restored, context='Restored clean image'))
export_image_to_fits(restored, '%s/imaging-dask_ical_restored.fits'
                     %(results_dir))

print(qa_image(residual[0], context='Residual clean image'))
export_image_to_fits(residual[0], '%s/imaging-dask_ical_residual.fits'
                     %(results_dir))


kill_dask_Client(c)

exit()
