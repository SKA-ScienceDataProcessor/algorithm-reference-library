
# coding: utf-8

# # Pipeline processing using serial workflows.
# 
# This notebook demonstrates the continuum imaging and ICAL pipelines. These are based on ARL functions wrapped up as SDP workflows using the serial class.

# In[1]:


#get_ipython().run_line_magic('matplotlib', 'inline')

import os
import sys

sys.path.append(os.path.join('..', '..'))

from data_models.parameters import arl_path

#results_dir = arl_path('test_results')
results_dir = './results/orig'

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
from wrappers.serial.visibility.coalesce import convert_blockvisibility_to_visibility,     convert_visibility_to_blockvisibility
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
import time
pp = pprint.PrettyPrinter()

import logging

def init_logging():
    log = logging.getLogger()
    logging.basicConfig(filename='%s/imaging-pipeline.log' % results_dir,
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)
    return log

log = init_logging()
log.info("Starting imaging-pipeline")


# In[2]:


#pylab.rcParams['figure.figsize'] = (12.0, 12.0)
#pylab.rcParams['image.cmap'] = 'Greys'


# We make the visibility. The parameter rmax determines the distance of the furthest antenna/stations used. All over parameters are determined from this number.

# In[3]:


nfreqwin=7
ntimes=5
rmax=300.0
frequency=numpy.linspace(1.0e8,1.2e8,nfreqwin)
#ntimes=11
#rmax=300.0
#frequency=numpy.linspace(0.9e8,1.1e8,nfreqwin)
channel_bandwidth=numpy.array(nfreqwin*[frequency[1]-frequency[0]])
times = numpy.linspace(-numpy.pi/3.0, numpy.pi/3.0, ntimes)
#phasecentre=SkyCoord(ra=+30.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')
phasecentre=SkyCoord(ra=+0.0 * u.deg, dec=-40.0 * u.deg, frame='icrs', equinox='J2000')

bvis_list=simulate_list_serial_workflow('LOWBD2',
                                         frequency=frequency, 
                                         channel_bandwidth=channel_bandwidth,
                                         times=times,
                                         phasecentre=phasecentre,
                                         order='frequency',
                                        rmax=rmax)

vis_list = [convert_blockvisibility_to_visibility(bv) for bv in bvis_list]
print('%d elements in vis_list' % len(vis_list))
log.debug('%d elements in vis_list' % len(vis_list))

# In[4]:


wprojection_planes=1
advice_low=advise_wide_field(vis_list[0], guard_band_image=8.0, delA=0.02,
                             wprojection_planes=wprojection_planes)

advice_high=advise_wide_field(vis_list[-1], guard_band_image=8.0, delA=0.02,
                              wprojection_planes=wprojection_planes)

vis_slices = advice_low['vis_slices']
npixel=advice_high['npixels2']
cellsize=min(advice_low['cellsize'], advice_high['cellsize'])


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


log.info('About to run predict to get predicted visibility')
log.info('About to run predict to get predicted visibility')
start=time.time()
predicted_vislist = predict_list_serial_workflow(vis_list, gleam_model,  
                                                context='wstack', vis_slices=vis_slices)
#log.info('About to run corrupt to get corrupted visibility')
#corrupted_vislist = corrupt_list_serial_workflow(predicted_vislist, phase_error=1.0)

end=time.time()
print('predict finished in %f seconds'%(end-start),flush=True)

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

start=time.time()
print('About to start invert' ,flush=True)

dirty_list = invert_list_serial_workflow(predicted_vislist, model_list, 
                                  context='wstack',
                                  vis_slices=vis_slices, dopsf=False)
psf_list = invert_list_serial_workflow(predicted_vislist, model_list, 
                                context='wstack',
                                vis_slices=vis_slices, dopsf=True)

end=time.time()
print('invert finished in %f seconds'%(end-start),flush=True)

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


# Now deconvolve using msclean

# In[ ]:


log.info('About to run deconvolve')
start=time.time()

deconvolved, _ =     deconvolve_list_serial_workflow(dirty_list, psf_list, model_imagelist=model_list, 
                            deconvolve_facets=8, deconvolve_overlap=16, deconvolve_taper='tukey',
                            scales=[0, 3, 10],
                            algorithm='msclean', niter=1000, 
                            fractional_threshold=0.1,
                            threshold=0.1, gain=0.1, psf_support=64)
    
end=time.time()
print('deconvolve finished in %f seconds'%(end-start),flush=True)
#show_image(deconvolved[0], cm='Greys', vmax=0.1, vmin=-0.01)
#plt.show()


# In[ ]:

log.info('About to run continuum imaging')
start=time.time()
continuum_imaging_list =     continuum_imaging_list_serial_workflow(predicted_vislist, 
                                            model_imagelist=model_list, 
                                            context='wstack', vis_slices=vis_slices, 
                                            scales=[0, 3, 10], algorithm='mmclean', 
                                            nmoment=3, niter=1000, 
                                            fractional_threshold=0.1,
                                            threshold=0.1, nmajor=5, gain=0.25,
                                            deconvolve_facets = 8, deconvolve_overlap=16, 
                                            deconvolve_taper='tukey', psf_support=64)


# In[ ]:

end=time.time()
print('continuum imaging finished in %f seconds'%(end-start),flush=True)


deconvolved = continuum_imaging_list[0][0]
residual = continuum_imaging_list[1][0]
restored = continuum_imaging_list[2][0]

#f=show_image(deconvolved, title='Clean image - no selfcal', cm='Greys', 
#             vmax=0.1, vmin=-0.01)
print(qa_image(deconvolved, context='Clean image - no selfcal'))

#plt.show()

#f=show_image(restored, title='Restored clean image - no selfcal', 
#             cm='Greys', vmax=1.0, vmin=-0.1)
print(qa_image(restored, context='Restored clean image - no selfcal'))
#plt.show()
export_image_to_fits(restored, '%s/imaging-dask_continuum_imaging_restored.fits' 
                     %(results_dir))

#f=show_image(residual[0], title='Residual clean image - no selfcal', cm='Greys', 
#             vmax=0.1, vmin=-0.01)
print(qa_image(residual[0], context='Residual clean image - no selfcal'))
#plt.show()
export_image_to_fits(residual[0], '%s/imaging-dask_continuum_imaging_residual.fits' 
                     %(results_dir))


# In[ ]:


#for chan in range(nfreqwin):
#    residual = continuum_imaging_list[1][chan]
    #show_image(residual[0], title='Channel %d' % chan, cm='Greys', 
    #           vmax=0.1, vmin=-0.01)
    #plt.show()


# In[ ]:


controls = create_calibration_controls()
        
controls['T']['first_selfcal'] = 1
controls['G']['first_selfcal'] = 3
controls['B']['first_selfcal'] = 4

controls['T']['timescale'] = 'auto'
controls['G']['timescale'] = 'auto'
controls['B']['timescale'] = 1e5

pp.pprint(controls)


# In[ ]:
start=time.time()
log.info('About to run ical')

# TODO I change this to predicted_vislist to make it deterministic, I hope it makes
# sense :)
#ical_list = ical_list_serial_workflow(corrupted_vislist, 
ical_list = ical_list_serial_workflow(predicted_vislist, 
                                        model_imagelist=model_list,  
                                        context='wstack', 
                                        calibration_context = 'TG', 
                                        controls=controls,
                                        scales=[0, 3, 10], algorithm='mmclean', 
                                        nmoment=3, niter=1000, 
                                        fractional_threshold=0.1,
                                        threshold=0.1, nmajor=5, gain=0.25,
                                        deconvolve_facets = 8, 
                                        deconvolve_overlap=16,
                                        deconvolve_taper='tukey',
                                        vis_slices=ntimes,
                                        timeslice='auto',
                                        global_solution=False, 
                                        psf_support=64,
                                        do_selfcal=True)


# In[ ]:

end=time.time()
print('ical finished in %f seconds'%(end-start),flush=True)

deconvolved = ical_list[0][0]
residual = ical_list[1][0]
restored = ical_list[2][0]

#f=show_image(deconvolved, title='Clean image', cm='Greys', vmax=1.0, vmin=-0.1)
print(qa_image(deconvolved, context='Clean image'))
#plt.show()

#f=show_image(restored, title='Restored clean image', cm='Greys', vmax=1.0, 
#             vmin=-0.1)
print(qa_image(restored, context='Restored clean image'))
#plt.show()
export_image_to_fits(restored, '%s/imaging-dask_ical_restored.fits' 
                     %(results_dir))

#f=show_image(residual[0], title='Residual clean image', cm='Greys', 
#             vmax=0.1, vmin=-0.01)
print(qa_image(residual[0], context='Residual clean image'))
#plt.show()
export_image_to_fits(residual[0], '%s/imaging-dask_ical_residual.fits' 
                     %(results_dir))

