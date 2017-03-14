import os
import sys

from time import clock

sys.path.append(os.path.join('..', '..'))

from matplotlib import pylab

pylab.rcParams['agg.path.chunksize'] = 10000
pylab.rcParams['figure.figsize'] = (12.0, 12.0)
pylab.rcParams['image.cmap'] = 'rainbow'

import numpy

from astropy.convolution import Gaussian2DKernel, convolve

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy import constants as const
from astropy.wcs.utils import pixel_to_skycoord

from matplotlib import pyplot as plt
from matplotlib.pyplot import cm 

from arl.visibility.operations import create_blockvisibility, vis_summary, copy_visibility
from arl.skycomponent.operations import create_skycomponent, insert_skycomponent, apply_beam_to_skycomponent
from arl.image.operations import show_image, export_image_to_fits, import_image_from_fits, qa_image,     create_image_from_array, copy_image, smooth_image
from arl.fourier_transforms.ftprocessor_base import create_image_from_visibility, predict_skycomponent_visibility 
from arl.fourier_transforms.fft_support import extract_mid
from arl.visibility.coalesce import coalesce_visibility, decoalesce_visibility
from arl.image.iterators import raster_iter
from arl.visibility.iterators import vis_timeslice_iter
from arl.util.testing_support import create_named_configuration, create_low_test_image, create_low_test_beam,     create_low_test_image_from_gleam, create_low_test_skycomponents_from_gleam
from arl.fourier_transforms.ftprocessor import *

import logging
log = logging.getLogger()
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))


# Construct the configuration and fill in the appropriate sampling values

# In[5]:

config = 'core'
if config == 'full':
    low = create_named_configuration('LOWBD2')
    b = 8e4
    cellsize = 0.00001
    npixel=8192
    nsnapshots = 30

else:
    low = create_named_configuration('LOWBD2-CORE')
    b = 4e3
    cellsize = 0.001
    npixel=256
    nsnapshots = 300

    
oversampling = 32

sampling_time = 35.0 / (oversampling * b)
log.info("Critical sampling time = %.5f (radians) %.2f (seconds)" % 
         (sampling_time, sampling_time * 43200.0 / numpy.pi))
sampling_frequency = 1e8 * 35.0 / (oversampling * b) 
log.info("Critical sampling frequency = %.5f (Hz) " % (sampling_frequency))
times = numpy.arange(0.0, + nsnapshots * sampling_time, sampling_time)
frequency = numpy.linspace(1e8 - sampling_frequency, 1e8 + sampling_frequency, 3)
channel_bandwidth = numpy.full_like(frequency, sampling_frequency)

log.info("Observing frequencies %s Hz" % (frequency))

log.info("Cellsize = %.6f radians" % (cellsize))


# We create the visibility holding the vis, uvw, time, antenna1, antenna2, weight columns in a table. The actual visibility values are zero.

# In[6]:

phasecentre = SkyCoord(ra=+0.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox=2000.0)
vt = create_blockvisibility(low, times, frequency, channel_bandwidth=channel_bandwidth,
                       weight=1.0, phasecentre=phasecentre, polarisation_frame=PolarisationFrame('stokesI'))


# Create components from GLEAM

# In[7]:

comps = create_low_test_skycomponents_from_gleam(flux_limit=1.0, polarisation_frame=PolarisationFrame("stokesI"),
                                             frequency=frequency, phasecentre=phasecentre)


# Create the low beam and apply it to the components, only those within the field of view.

# In[8]:

model = create_image_from_visibility(vt, npixel=npixel, cellsize=cellsize, frequency=frequency,
                                     polarisation_frame=PolarisationFrame('stokesI'),
                                     phasecentre=phasecentre, nchan=len(frequency))
beam=create_low_test_beam(model)
comps = apply_beam_to_skycomponent(comps, beam)
model = insert_skycomponent(model, comps)

# In[10]:

vt.data['uvw'][:,2] = 0
vt = predict_skycomponent_blockvisibility(vt, comps)


# Now we coalesce the data

# In[11]:

coalescence_factor=1.0

cvt, cindex = coalesce_visibility(vt, coalescence_factor=coalescence_factor, max_coalescence=10)
