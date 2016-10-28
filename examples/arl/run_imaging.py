# This script works through the high level arl interface to crocodile, making a fake data set and then
# deconvolving it. Finally the full and residual visibility are plotted.

import sys

sys.path.append('../..')
sys.path.append('..')

##### Setup logging, matplotlib

import logging

log = logging.getLogger("run_imaging")
logging.basicConfig(filename='./run_imaging.log', level=logging.DEBUG)
# define a new Handler to log to console as well
console = logging.StreamHandler()
logging.getLogger('').addHandler(console)

import numpy
import matplotlib

matplotlib.use('PDF')
from matplotlib import pyplot as plt

import pylab

pylab.rcParams['figure.figsize'] = (12.0, 12.0)
pylab.rcParams['image.cmap'] = 'rainbow'

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy import constants as const

from arl.image.image_deconvolution import deconvolve_cube
from arl.visibility.visibility_operations import create_visibility, combine_visibility, aq_visibility
from arl.image.image_operations import show_image, export_image_to_fits
from arl.util.testing_support import create_named_configuration, create_test_image
from arl.fourier_transforms.ftprocessor import *

##### End of setup
cellsize = 2e-5
params = {'wstep': 100.0, 'npixel': 512, 'cellsize': cellsize, 'niter': 1000, 'scales':[0,3,10,30],
          'threshold': 0.001, 'fracthresh': 0.01, 'weighting':'uniform'}
doshow = True

log.info("run_imaging: Starting")

# We construct a VLA configuration and then shrink it to match our test image.
vlaa = create_named_configuration('VLAA')

# We create the visibility. This just makes the uvw, time, antenna1, antenna2, weight columns in a table
vlaa.data['xyz'] = vlaa.data['xyz']
times = numpy.arange(-numpy.pi / 2.0, +numpy.pi / 2.0, 0.05)
frequency = numpy.array([1e8])
reffrequency = numpy.max(frequency)
phasecentre = SkyCoord(0.0 * u.rad, u.rad * numpy.pi / 4, frame='icrs', equinox=2000.0)
vis = create_visibility(vlaa, times, frequency, weight=1.0, phasecentre=phasecentre)

# Plot the synthesized uv coverage, including for MFS
if doshow:
    plt.clf()
    for f in frequency:
        x = f / const.c
        plt.plot(x * vis.data['uvw'][:, 0], x * vis.data['uvw'][:, 1], '.', color='b')
        plt.plot(-x * vis.data['uvw'][:, 0], -x * vis.data['uvw'][:, 1], '.', color='r')

# Read the venerable test image, constructing an image, and catch the axes for our purposes
m31image = create_test_image(cellsize=cellsize)

# Now we can predict_visibility the visibility from this skymodel
vis = predict_2d(vis, m31image, params=params)

# To check that we got the prediction right, plot the amplitude of the visibility.
if doshow:
    uvdist = numpy.sqrt(vis.data['uvw'][:, 0] ** 2 + vis.data['uvw'][:, 1] ** 2)
    plt.clf()
    plt.plot(uvdist, numpy.abs(vis.data['vis'][:, 0, 0]), '.')
    plt.xlabel('uvdist')
    plt.ylabel('Amp Visibility')
    plt.show()

# Make the dirty image and point spread function. They are empty at this point
dirty = create_image_from_visibility (vis, params)
psf =   create_image_from_visibility (vis, params)

# Reweight the data
vis = weight_visibility(vis, dirty, params)

psf = invert_2d(vis, psf, dopsf=True, params=params)
show_image(psf)
log.info("run_imaging: Max, min in PSF         = %.6f, %.6f" % (psf.data.max(), psf.data.min()))


# Now use straightforward 2D transform to make dirty image and psf.
dirty = invert_2d(vis, dirty, params=params)

psfmax = numpy.max(psf.data)
dirty.data=dirty.data/psfmax
psf.data=psf.data/psfmax

log.info("run_imaging: Max, min in dirty image = %.6f, %.6f" % (dirty.data.max(), dirty.data.min()))

export_image_to_fits(dirty, 'run_imaging_dirty.fits')
export_image_to_fits(psf, 'run_imaging_psf.fits')

# Deconvolve using clean
cleanimage, residual = deconvolve_cube(dirty, psf, params=params)

# Show the results
fig = show_image(cleanimage)
fig = show_image(residual)
export_image_to_fits(cleanimage, 'run_imaging_cleanimage.fits')
export_image_to_fits(residual, 'run_imaging_residual.fits')

# Predict the visibility of the model
vismodel = create_visibility(vlaa, times, frequency, weight=1.0, phasecentre=phasecentre)
vismodel = predict_2d(vismodel, cleanimage, params=params)
visres = combine_visibility(vis, vismodel, 1.0, -1.0)

# Now we will plot the original visibility and the residual visibility.

if doshow:
    uvdist = numpy.sqrt(vis.data['uvw'][:, 0] ** 2 + vis.data['uvw'][:, 1] ** 2)
    plt.clf()
    plt.plot(uvdist, numpy.abs(vis.data['vis'][:, 0, 0]), '.', color='b')
    plt.plot(uvdist, numpy.abs(vis.data['vis'][:, 0, 0] - vismodel.data['vis'][:, 0, 0]), '.', color='r')
    plt.xlabel('uvdist')
    plt.ylabel('Amp Visibility')
    plt.show()

qa = aq_visibility(vis)
qares = aq_visibility(visres)
log.info("Observed visibility: %s" % (str(qa.data)))
log.info("Residual visibility: %s" % (str(qares.data)))

log.info("run_imaging: End of processing")
