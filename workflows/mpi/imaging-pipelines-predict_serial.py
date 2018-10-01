
# coding: utf-8

# # Pipeline processing using serial workflows.
# 
# This is a serial unrolled version of the predict step

# In[1]:


#get_ipython().run_line_magic('matplotlib', 'inline')

import os
import sys

sys.path.append(os.path.join('..', '..'))

from data_models.parameters import arl_path

results_dir = './results'

#from matplotlib import pylab

#pylab.rcParams['figure.figsize'] = (12.0, 12.0)
#pylab.rcParams['image.cmap'] = 'rainbow'

import numpy

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs.utils import pixel_to_skycoord

#from matplotlib import pyplot as plt

from data_models.polarisation import PolarisationFrame

from wrappers.serial.calibration.calibration import solve_gaintable
from wrappers.serial.calibration.operations import apply_gaintable
from wrappers.serial.calibration.calibration_control import create_calibration_controls
from wrappers.serial.visibility.base import create_blockvisibility
from wrappers.serial.skycomponent.operations import create_skycomponent
from wrappers.serial.image.deconvolution import deconvolve_cube
#from wrappers.serial.image.operations import show_image, export_image_to_fits, qa_image
from wrappers.serial.image.operations import export_image_to_fits, qa_image
from wrappers.serial.visibility.iterators import vis_timeslice_iter
from wrappers.serial.simulation.testing_support import create_named_configuration, create_low_test_image_from_gleam
from wrappers.serial.imaging.base import predict_2d, create_image_from_visibility, advise_wide_field

from workflows.serial.imaging.imaging_serial import invert_list_serial_workflow,     predict_list_serial_workflow, deconvolve_list_serial_workflow
from workflows.serial.simulation.simulation_serial import simulate_list_serial_workflow,     corrupt_list_serial_workflow
from workflows.serial.pipelines.pipeline_serial import continuum_imaging_list_serial_workflow,     ical_list_serial_workflow

import pprint

pp = pprint.PrettyPrinter()

import logging

def init_logging():
    log = logging.getLogger()
    logging.basicConfig(filename='%s/imaging-predict.log' % results_dir,
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)
log = logging.getLogger()
logging.info("Starting imaging-pipeline")


# In[2]:


#pylab.rcParams['figure.figsize'] = (12.0, 12.0)
#pylab.rcParams['image.cmap'] = 'Greys'


# We make the visibility. The parameter rmax determines the distance of the furthest antenna/stations used. All over parameters are determined from this number.

# In[3]:


nfreqwin=7
ntimes=11
rmax=300.0
frequency=numpy.linspace(0.9e8,1.1e8,nfreqwin)
channel_bandwidth=numpy.array(nfreqwin*[frequency[1]-frequency[0]])
times = numpy.linspace(-numpy.pi/3.0, numpy.pi/3.0, ntimes)
phasecentre=SkyCoord(ra=+30.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')

vis_list=simulate_list_serial_workflow('LOWBD2',
                                         frequency=frequency, 
                                         channel_bandwidth=channel_bandwidth,
                                         times=times,
                                         phasecentre=phasecentre,
                                         order='frequency',
                                        rmax=rmax)
print('%d elements in vis_list' % len(vis_list))


# In[4]:


wprojection_planes=1
advice_low=advise_wide_field(vis_list[0], guard_band_image=8.0, delA=0.02,
                             wprojection_planes=wprojection_planes)

advice_high=advise_wide_field(vis_list[-1], guard_band_image=8.0, delA=0.02,
                              wprojection_planes=wprojection_planes)

vis_slices = advice_low['vis_slices']
npixel=advice_high['npixels2']
cellsize=min(advice_low['cellsize'], advice_high['cellsize'])
print('After advice: vis_slices %d npixel %d cellsize %d' % (vis_slices, npixel, cellsize))

# Now make a graph to fill with a model drawn from GLEAM 

# In[ ]:


gleam_model = [create_low_test_image_from_gleam(npixel=npixel,
                                                               frequency=[frequency[f]],
                                                               channel_bandwidth=[channel_bandwidth[f]],
                                                               cellsize=cellsize,
                                                               phasecentre=phasecentre,
                                                               polarisation_frame=PolarisationFrame("stokesI"),
                                                               flux_limit=1.0,
                                                               applybeam=True)
                     for f, freq in enumerate(frequency)]
log.info('About to make GLEAM model')


# In[ ]:

original_predict=True
if original_predict:
    log.info('About to run predict to get predicted visibility')
    predicted_vislist = predict_list_serial_workflow(vis_list, gleam_model,  
                                                context='wstack', vis_slices=vis_slices)
else:
    log.info('About to run predict to get predicted visibility')
    model_imagelist=gleam_model
    context='wstack'
    facets=1
    
    assert len(vis_list) == len(model_imagelist), "Model must be the same length as the vis_list"
    from workflows.shared.imaging.imaging_shared import imaging_context
    from processing_components.image.gather_scatter import image_scatter_facets, image_gather_facets
    from processing_components.visibility.gather_scatter import visibility_scatter, visibility_gather
    from workflows.shared.imaging.imaging_shared import sum_invert_results, remove_sumwt, sum_predict_results, \
		        threshold_list

    c = imaging_context(context)
    vis_iter = c['vis_iterator']
    predict = c['predict']
    
    def predict_ignore_none(vis, model):
        if vis is not None:
            return predict(vis, model, context=context, facets=facets, vis_slices=vis_slices)
        else:
            return None
    
    image_results_list_list = list()
    # Loop over all frequency windows
    for freqwin, vis_lst in enumerate(vis_list):
        # Create the graph to divide an image into facets. This is by reference.
        facet_lists = image_scatter_facets(model_imagelist[freqwin], facets=facets)
        # Create the graph to divide the visibility into slices. This is by copy.
        sub_vis_lists = visibility_scatter(vis_lst, vis_iter, vis_slices)
        facet_vis_lists = list()
        # Loop over sub visibility
        for sub_vis_list in sub_vis_lists:
            facet_vis_results = list()
            # Loop over facets
            for facet_list in facet_lists:
                # Predict visibility for this subvisibility from this facet
                facet_vis_list = predict_ignore_none(sub_vis_list, facet_list)
                facet_vis_results.append(facet_vis_list)
            # Sum the current sub-visibility over all facets
            facet_vis_lists.append(sum_predict_results(facet_vis_results))
        # Sum all sub-visibilties
        image_results_list_list.append(visibility_gather(facet_vis_lists, vis_lst, vis_iter))
    
    predicted_vislist=image_results_list_list

#log.info('About to run corrupt to get corrupted visibility')
#corrupted_vislist = corrupt_list_serial_workflow(predicted_vislist, phase_error=1.0)


# Get the LSM. This is currently blank.

# In[ ]:


model_list = [create_image_from_visibility(vis_list[f],
                                                     npixel=npixel,
                                                     frequency=[frequency[f]],
                                                     channel_bandwidth=[channel_bandwidth[f]],
                                                     cellsize=cellsize,
                                                     phasecentre=phasecentre,
                                                     polarisation_frame=PolarisationFrame("stokesI"))
               for f, freq in enumerate(frequency)]


# In[ ]:


dirty_list = invert_list_serial_workflow(predicted_vislist, model_list, 
                                  context='wstack',
                                  vis_slices=vis_slices, dopsf=False)
psf_list = invert_list_serial_workflow(predicted_vislist, model_list, 
                                context='wstack',
                                vis_slices=vis_slices, dopsf=True)


# Create and execute graphs to make the dirty image and PSF

# In[ ]:


log.info('About to run invert to get dirty image')
dirty = dirty_list[0][0]
#show_image(dirty, cm='Greys', vmax=1.0, vmin=-0.1)
#plt.show()
print(qa_image(dirty))
export_image_to_fits(dirty, '%s/imaging-dirty.fits' 
                     %(results_dir))

log.info('About to run invert to get PSF')
psf = psf_list[0][0]
#show_image(psf, cm='Greys', vmax=0.1, vmin=-0.01)
#plt.show()
print(qa_image(psf))
export_image_to_fits(psf, '%s/imaging-psf.fits' 
                     %(results_dir))

