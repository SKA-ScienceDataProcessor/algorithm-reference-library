

# This script works through the high level arl interface to crocodile, making a fake data set and then
# deconvolving it. Finally the full and residual visibility are plotted.

import sys, os
sys.path.append('../..') 
print(os.getcwd())

import sys
sys.stdout = open('%s.txt'%(sys.argv[0]), 'w')

import pylab
pylab.rcParams['figure.figsize'] = (12.0, 12.0)
pylab.rcParams['image.cmap'] = 'rainbow'

from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel, pixel_to_skycoord
from astropy import units as u

from matplotlib import pyplot as plt

from arl.image_deconvolution import deconvolve_cube
from arl.visibility_operations import create_visibility
from arl.fourier_transform import *
from arl.skymodel_operations import create_skymodel_from_image, add_component_to_skymodel, create_skycomponent, find_point_source
from arl.image_operations import show_image, create_image_from_fits, save_image_to_fits, replicate_image
from arl.visibility_simulation import filter_configuration, create_named_configuration


# We construct a VLA configuration and then shrink it to match our test image.

kwargs = {}

vlaa = filter_configuration(create_named_configuration('VLAA'), **kwargs)
vlaa.data['xyz']=vlaa.data['xyz']/10.0


# We create the visibility. This just makes the uvw, time, antenna1, antenna2, weight columns in a table

times = numpy.arange(-numpy.pi/2.0, +numpy.pi/2.0,0.05)
frequency = numpy.array([1e8])

reffrequency = numpy.max(frequency)
phasecentre = SkyCoord(0.0*u.rad, u.rad*numpy.pi/4, frame='icrs', equinox=2000.0)
vt = create_visibility(vlaa, times, frequency, weight=1.0, phasecentre=phasecentre)


# Plot the synthesized uv coverage, including for MFS

plt.clf()
for f in frequency:
    x=f/const.c
    plt.plot(x*vt.data['uvw'][:,0], x*vt.data['uvw'][:,1], '.', color='b')
    plt.plot(-x*vt.data['uvw'][:,0], -x*vt.data['uvw'][:,1], '.', color='r')


# Read the venerable test image, constructing an image

m31image = create_image_from_fits("./data/models/M31.MOD")
fig = plt.figure()
cellsize=180.0*0.0001/numpy.pi
m31image.wcs.wcs.cdelt[0]=-cellsize
m31image.wcs.wcs.cdelt[1]=+cellsize
m31image.wcs.wcs.radesys='ICRS'
m31image.wcs.wcs.equinox=2000.00

# Show the model image
fig.add_subplot(111, projection=m31image.wcs)
plt.imshow(m31image.data, origin='lower', cmap='rainbow')
plt.xlabel('RA---SIN')
plt.ylabel('DEC--SIN')
plt.show()

# This image is only 2 dimensional. We need extra axes frequency and stokes.

m31image4D=replicate_image(m31image, shape=[1, 1, 4, len(frequency)])
m31sm = create_skymodel_from_image(m31image4D)

# We need a linear reference frame to inset a model source. This is a bit involved die to the Astropy way of doing
# things
wall = m31image.wcs
wall.wcs.radesys='ICRS'
wall.wcs.equinox=2000.00
print(wall.wcs.radesys)
print(wall.wcs.equinox)
sc=pixel_to_skycoord(128, 128, wall, 1, 'wcs')
compabsdirection=SkyCoord("-1.0d", "37.0d", frame='icrs', equinox=2000.0)
pixloc = skycoord_to_pixel(compabsdirection, wall, 1)
scrt = pixel_to_skycoord(pixloc[0], pixloc[1], wall, 1, 'wcs')
sof=sc.skyoffset_frame()
compreldirection = compabsdirection.transform_to(sof)

# Create a skycomponent and add it to the skymodel
comp1= create_skycomponent(flux=numpy.array([[1.0, 0.0, 0.0, 0.0]]), frequency=frequency, direction=compreldirection)
m31sm=add_component_to_skymodel(m31sm, comp1)

# Now we can predict_visibility the visibility from this skymodel
kwargs={'wstep':100.0, 'npixel':256, 'cellsize':0.0001}
vt = predict_visibility(vt, m31sm, **kwargs)

# To check that we got the prediction right, plot the amplitude of the visibility.
uvdist=numpy.sqrt(vt.data['uvw'][:,0]**2+vt.data['uvw'][:,1]**2)
plt.clf()
plt.plot(uvdist, numpy.abs(vt.data['vis'][:,0,0]), '.')
plt.xlabel('uvdist')
plt.ylabel('Amp Visibility')
plt.show()


# Make the dirty image and point spread function

kwargs={}
kwargs['npixel']=512
kwargs['cellsize']=0.0001
kwargs['wstep']=30.0
dirty, psf, sumwt = invert_visibility(vt, **kwargs)
show_image(dirty)
print("Max, min in dirty image = %.6f, %.6f, sum of weights = %f" % (dirty.data.max(), dirty.data.min(), sumwt))

print("Max, min in PSF         = %.6f, %.6f, sum of weights = %f" % (psf.data.max(), psf.data.min(), sumwt))

save_image_to_fits(dirty, 'dirty.fits')
save_image_to_fits(psf, 'psf.fits')
m31compnew = find_point_source(dirty)


# Deconvolve using clean

kwargs={'niter':100, 'threshold':0.001, 'fracthresh':0.01}
comp, residual = deconvolve_cube(dirty, psf, **kwargs)

# Show the results

fig=show_image(comp)
fig=show_image(residual)


# Predict the visibility of the model

kwargs={'wstep':30.0}
vt = predict_visibility(vt, m31sm, **kwargs)
modelsm=create_skymodel_from_image(comp)
vtmodel = create_visibility(vlaa, times, frequency, weight=1.0, phasecentre=phasecentre)
vtmodel.data = vt.data.copy()
vtmodel=predict_visibility(vtmodel, modelsm, **kwargs)


# Now we will plot the original visibility and the residual visibility.

uvdist=numpy.sqrt(vt.data['uvw'][:,0]**2+vt.data['uvw'][:,1]**2)
plt.clf()
plt.plot(uvdist, numpy.abs(vt.data['vis'][:,0,0]), '.', color='b')
plt.plot(uvdist, numpy.abs(vt.data['vis'][:,0,0]-vtmodel.data['vis'][:,0,0]), '.', color='r')
plt.xlabel('uvdist')
plt.ylabel('Amp Visibility')
plt.show()
