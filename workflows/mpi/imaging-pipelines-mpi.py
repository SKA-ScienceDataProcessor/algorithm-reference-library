
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
from mpi4py import MPI

results_dir = './results/mpi'

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

from workflows.mpi.pipelines.pipeline_mpi import continuum_imaging_list_mpi_workflow
from workflows.mpi.imaging.imaging_mpi import predict_list_mpi_workflow, invert_list_mpi_workflow, deconvolve_list_mpi_workflow

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

# ################### Rationale of data distribution: ################### #
# In this version all data resides at rank0 and needs to be distributed   #
# at every function when needed.                                          #
# TODO: Pass on the comm parameter!
# vis_list -> rank0                                                       #
# vis_slices, npixel, cellsize -> rep                                     #
# gleam_model -> rank0 (later rep)                                        #
# predicted_vis -> rank0 (later dist)                                     #
# model_list ->rank0 (later rep)
# disrty_list psf_list -> rank0 (later dist)
# continuum_imaging_list -> rank0
# ####################################################################### #




#pylab.rcParams['figure.figsize'] = (12.0, 12.0)
#pylab.rcParams['image.cmap'] = 'Greys'


# Set up MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# We make the visibility. The parameter rmax determines the distance of the furthest antenna/stations used. All over parameters are determined from this number.

# In[3]:


nfreqwin=7
ntimes=11
rmax=300.0
frequency=numpy.linspace(0.9e8,1.1e8,nfreqwin)
channel_bandwidth=numpy.array(nfreqwin*[frequency[1]-frequency[0]])
times = numpy.linspace(-numpy.pi/3.0, numpy.pi/3.0, ntimes)
phasecentre=SkyCoord(ra=+30.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')

print('frequency len %d frequency list:' % len(frequency),flush=True)
print(frequency)

if rank == 0:
    vis_list=simulate_list_serial_workflow('LOWBD2',
                                         frequency=frequency, 
                                         channel_bandwidth=channel_bandwidth,
                                         times=times,
                                         phasecentre=phasecentre,
                                         order='frequency',
                                        rmax=rmax)
else:
    vis_list=list()

print('%d: %d elements in vis_list' % (rank,len(vis_list)),flush=True)
print(vis_list)


# In[4]:

if rank == 0:
    wprojection_planes=1
    advice_low=advise_wide_field(vis_list[0], guard_band_image=8.0, delA=0.02,
                             wprojection_planes=wprojection_planes)

    advice_high=advise_wide_field(vis_list[-1], guard_band_image=8.0, delA=0.02,
                              wprojection_planes=wprojection_planes)

    vis_slices = advice_low['vis_slices']
    npixel=advice_high['npixels2']
    cellsize=min(advice_low['cellsize'], advice_high['cellsize'])

else:
    vis_slices = 0
    npixel = 0
    cellsize = 0

(vis_slices,npixel,cellsize) = comm.bcast((vis_slices,npixel,cellsize),root=0)
print('%d: After advice: vis_slices %d npixel %d cellsize %d' %
      (rank,vis_slices, npixel, cellsize),flush=True)

# Now make a graph to fill with a model drawn from GLEAM 

# In[ ]:

sub_frequency = numpy.array_split(frequency, size)
sub_channel_bandwidth = numpy.array_split(channel_bandwidth,size)

sub_gleam_model = [create_low_test_image_from_gleam(npixel=npixel,
                                                               frequency=[sub_frequency[rank][f]],
                                                               channel_bandwidth=[sub_channel_bandwidth[rank][f]],
                                                               cellsize=cellsize,
                                                               phasecentre=phasecentre,
                                                               polarisation_frame=PolarisationFrame("stokesI"),
                                                               flux_limit=1.0,
                                                               applybeam=True)
                     for f, freq in enumerate(sub_frequency[rank])]

# NOTE: We could do an allgather here to avoid bcast of
# each freqw during predict, it would safe time but use more space

gleam_model=comm.gather(sub_gleam_model,root=0)
if rank==0:
    gleam_model=numpy.concatenate(gleam_model)
else:
    gleam_model=list()

# In[ ]:
log.info('About to make GLEAM model')

original_predict=False
if original_predict:
    if rank==0:
        log.info('About to run predict to get predicted visibility')
        predicted_vislist = predict_list_serial_workflow(vis_list, gleam_model,
                                                context='wstack', vis_slices=vis_slices)
else:
    log.info('About to run predict to get predicted visibility')
    # All procs call the function but only rank=0 gets the predicted_vislist
    predicted_vislist = predict_list_mpi_workflow(vis_list, gleam_model,
                                                context='wstack',
                                                  vis_slices=vis_slices)

    #log.info('About to run corrupt to get corrupted visibility')
    #corrupted_vislist = corrupt_list_serial_workflow(predicted_vislist, phase_error=1.0)


    # Get the LSM. This is currently blank.

    # In[ ]:
    ### I need to scatter vis_list cause worker don't have it
    ## frequency and channel_bandwidth are replicated and they have already
    ## been split

sub_vis_list= numpy.array_split(vis_list, size)
sub_vis_list=comm.scatter(sub_vis_list,root=0)

sub_model_list = [create_image_from_visibility(sub_vis_list[f],
                                                     npixel=npixel,
                                                     frequency=[sub_frequency[rank][f]],
                                                     channel_bandwidth=[sub_channel_bandwidth[rank][f]],
                                                     cellsize=cellsize,
                                                     phasecentre=phasecentre,
                                                     polarisation_frame=PolarisationFrame("stokesI"))
               for f, freq in enumerate(sub_frequency[rank])]

# NOTE: We could do allgather here, if enough memory space
model_list=comm.gather(sub_model_list,root=0)
if rank==0:
    model_list=numpy.concatenate(model_list)
    # In[ ]:
else:
    model_list=list()

print('%d model_list len %d' %(rank,len(model_list)),flush=True)
print('%d: About to start invert'%(rank),flush=True)
original_invert=False
if original_invert:
    if rank==0:
        dirty_list = invert_list_serial_workflow(predicted_vislist, model_list, 
                                  context='wstack',
                                  vis_slices=vis_slices, dopsf=False)
        psf_list = invert_list_serial_workflow(predicted_vislist, model_list, 
                                context='wstack',
                                vis_slices=vis_slices, dopsf=True)
else:
    dirty_list = invert_list_mpi_workflow(predicted_vislist, model_list, 
                                  context='wstack',
                                  vis_slices=vis_slices, dopsf=False)
    psf_list = invert_list_mpi_workflow(predicted_vislist, model_list, 
                                context='wstack',
                                vis_slices=vis_slices, dopsf=True)


    # Create and execute graphs to make the dirty image and PSF

    # In[ ]:
        
if rank==0:
    print("sumwts",flush=True)
    print(dirty_list[0][1])

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
print('%d: About to run deconvolve'%(rank),flush=True)
original_deconv=False
if original_deconv:
    if rank==0:

        deconvolved, _ =     deconvolve_list_serial_workflow(dirty_list, psf_list, model_imagelist=model_list, 
                            deconvolve_facets=8, deconvolve_overlap=16, deconvolve_taper='tukey',
                            scales=[0, 3, 10],
                            algorithm='msclean', niter=1000, 
                            fractional_threshold=0.1,
                            threshold=0.1, gain=0.1, psf_support=64)
else:

    deconvolved, _ =     deconvolve_list_mpi_workflow(dirty_list, psf_list, model_imagelist=model_list, 
                            deconvolve_facets=8, deconvolve_overlap=16, deconvolve_taper='tukey',
                            scales=[0, 3, 10],
                            algorithm='msclean', niter=1000, 
                            fractional_threshold=0.1,
                            threshold=0.1, gain=0.1, psf_support=64)
    
#show_image(deconvolved[0], cm='Greys', vmax=0.1, vmin=-0.01)
#plt.show()


# In[ ]:

log.info('About to run continuum imaging')
print('%d: About to run continuum imaging'%(rank),flush=True)

original_continuumimaging=False
if original_continuumimaging:
    if rank==0:
        continuum_imaging_list =     continuum_imaging_list_serial_workflow(predicted_vislist, 
                                            model_imagelist=model_list, 
                                            context='wstack', vis_slices=vis_slices, 
                                            scales=[0, 3, 10], algorithm='mmclean', 
                                            nmoment=3, niter=1000, 
                                            fractional_threshold=0.1,
                                            threshold=0.1, nmajor=5, gain=0.25,
                                            deconvolve_facets = 8, deconvolve_overlap=16, 
                                            deconvolve_taper='tukey', psf_support=64)
else:
    continuum_imaging_list =     continuum_imaging_list_mpi_workflow(predicted_vislist, 
                                            model_imagelist=model_list, 
                                            context='wstack', vis_slices=vis_slices, 
                                            scales=[0, 3, 10], algorithm='mmclean', 
                                            nmoment=3, niter=1000, 
                                            fractional_threshold=0.1,
                                            threshold=0.1, nmajor=5, gain=0.25,
                                            deconvolve_facets = 8, deconvolve_overlap=16, 
                                            deconvolve_taper='tukey', psf_support=64)




# In[ ]:
log.info('About to run continuum imaging')

if rank==0:

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
