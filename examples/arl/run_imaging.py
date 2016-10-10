# This script works through the high level arl interface to crocodile, making a fake data set and then
# deconvolving it. Finally the full and residual visibility are plotted.

import sys, os

sys.path.append('../..')

##### Setup logging, matplotlib

import logging

log = logging.getLogger("run_imaging")
logging.basicConfig(filename='./run_imaging.log', level=logging.DEBUG)
# define a new Handler to log to console as well
console = logging.StreamHandler()
logging.getLogger('').addHandler(console)

import matplotlib

matplotlib.use('PDF')
from matplotlib import pyplot as plt

import pylab

pylab.rcParams['figure.figsize'] = (12.0, 12.0)
pylab.rcParams['image.cmap'] = 'rainbow'

from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel, pixel_to_skycoord
from astropy import units as u

from arl.image_deconvolution import deconvolve_cube
from arl.visibility_operations import create_visibility, combine_visibility, aq_visibility
from arl.fourier_transforms import *
from arl.skymodel_operations import create_skymodel_from_image, add_component_to_skymodel, create_skycomponent, \
    find_skycomponent
from arl.image_operations import show_image, export_image_to_fits
from arl.testing_support import create_named_configuration, create_test_image

##### End of setup
params = {}
doshow = False

log.info("run_imaging: Starting")

# We construct a VLA configuration and then shrink it to match our test image.
vlaa = create_named_configuration('VLAA')

# We create the visibility. This just makes the uvw, time, antenna1, antenna2, weight columns in a table
vlaa.data['xyz'] = vlaa.data['xyz'] / 10.0
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
m31image = create_test_image()
cellsize = 180.0 * 0.0001 / numpy.pi
m31image.wcs.wcs.cdelt[0] = -cellsize
m31image.wcs.wcs.cdelt[1] = +cellsize
m31image.wcs.wcs.radesys = 'ICRS'
m31image.wcs.wcs.equinox = 2000.00

# Show the model image
if doshow:
    plt.clf()
    plt.imshow(m31image.data, origin='lower', cmap='rainbow')
    plt.xlabel('RA---SIN')
    plt.ylabel('DEC--SIN')
    plt.show()

# We need a linear reference frame to inset a model source. This is a bit involved due to the Astropy way of doing
# things
wall = m31image.wcs
wall.wcs.radesys = 'ICRS'
wall.wcs.equinox = 2000.00
sc = pixel_to_skycoord(128, 128, wall, 1, 'wcs')
compabsdirection = SkyCoord("-1.0d", "37.0d", frame='icrs', equinox=2000.0)
pixloc = skycoord_to_pixel(compabsdirection, wall, 1)
scrt = pixel_to_skycoord(pixloc[0], pixloc[1], wall, 1, 'wcs')
sof = sc.skyoffset_frame()
compreldirection = compabsdirection.transform_to(sof)

# Create a skycomponent and add it to the skymodel
comp1 = create_skycomponent(flux=numpy.array([[30.0, 0.0, 0.0, 0.0]]), frequency=frequency,
                            direction=compreldirection)
m31sm = create_skymodel_from_image(m31image)
m31sm = add_component_to_skymodel(m31sm, comp1)

# Now we can predict_visibility the visibility from this skymodel
params = {'wstep': 100.0, 'npixel': 256, 'cellsize': 0.0001}
vis = predict_visibility(vis, m31sm, params=params)

# To check that we got the prediction right, plot the amplitude of the visibility.
if doshow:
    uvdist = numpy.sqrt(vis.data['uvw'][:, 0] ** 2 + vis.data['uvw'][:, 1] ** 2)
    plt.clf()
    plt.plot(uvdist, numpy.abs(vis.data['vis'][:, 0, 0]), '.')
    plt.xlabel('uvdist')
    plt.ylabel('Amp Visibility')
    plt.show()

# Make the dirty image and point spread function

params = {'wstep': 30.0, 'npixel': 512, 'cellsize': 0.0001}

dirty, psf, sumwt = invert_visibility(vis, params=params)
show_image(dirty)
log.info("run_imaging: Max, min in dirty image = %.6f, %.6f, sum of weights = %f" % (
dirty.data.max(), dirty.data.min(), sumwt))

log.info(
    "run_imaging: Max, min in PSF         = %.6f, %.6f, sum of weights = %f" % (psf.data.max(), psf.data.min(), sumwt))

export_image_to_fits(dirty, 'run_imaging_dirty.fits')
export_image_to_fits(psf, 'run_imaging_psf.fits')

m31compnew = find_skycomponent(dirty)

# Deconvolve using clean

params = {'niter': 1000, 'threshold': 0.001, 'fracthresh': 0.01}

comp, residual = deconvolve_cube(dirty, psf, params=params)

# Show the results

fig = show_image(comp)
fig = show_image(residual)

# Predict the visibility of the model

params = {'wstep': 30.0}

vis = predict_visibility(vis, m31sm, params=params)
modelsm = create_skymodel_from_image(comp)
vismodel = create_visibility(vlaa, times, frequency, weight=1.0, phasecentre=phasecentre)
vismodel.data = vis.data.copy()
vismodel = predict_visibility(vismodel, modelsm, params={})
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
