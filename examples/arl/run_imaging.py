
# coding: utf-8

# This script works through the high level arl interface to crocodile, making a fake data set and then
# deconvolving it. Finally the full and residual visibility are plotted.

import sys, os
sys.path.append('../..') 
print(os.getcwd())

import pylab
pylab.rcParams['figure.figsize'] = (12.0, 12.0)
pylab.rcParams['image.cmap'] = 'rainbow'

from astropy import constants as const
from astropy import units as u
from astropy.coordinates import SkyCoord, CartesianRepresentation, SkyOffsetFrame
import scipy.special

from matplotlib import pylab
from matplotlib import pyplot as plt

from arl.clean import clean
from arl.visibility import create_visibility
from arl.imaging import *
from arl.skymodel import SkyModel, skymodel_from_image, skymodel_add_image, skymodel_add_component
from arl.skycomponent import *
from arl.image import image_show, image_from_fits, image_to_fits, image_replicate, fitcomponent
from arl.configuration import configuration_filter, named_configuration


# We construct a VLA configuration and then shrink it to match our test image.

kwargs = {}

vlaa = configuration_filter(named_configuration('VLAA'), **kwargs)
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

m31image = image_from_fits("./data/models/M31.MOD")
fig = plt.figure()
cellsize=180.0*0.0001/numpy.pi
m31image.wcs.wcs.cdelt[0]=-cellsize
m31image.wcs.wcs.cdelt[1]=+cellsize
m31image.wcs.wcs.radesys='ICRS'
m31image.wcs.wcs.equinox=2000.00


fig.add_subplot(111, projection=m31image.wcs)
plt.imshow(m31image.data, origin='lower', cmap='rainbow')
plt.xlabel('RA---SIN')
plt.ylabel('DEC--SIN')
plt.show()


from astropy.wcs.utils import skycoord_to_pixel, pixel_to_skycoord
wall = m31image.wcs
wall.wcs.radesys='ICRS'
wall.wcs.equinox=2000.00
print(wall.wcs.radesys)
print(wall.wcs.equinox)
sc=pixel_to_skycoord(128, 128, wall, 1, 'wcs')
print(sc)
pixloc = skycoord_to_pixel(SkyCoord("-1.0d", "37.0d", frame='icrs', equinox=2000.0), wall, 1)
print(pixloc)
scrt = pixel_to_skycoord(pixloc[0], pixloc[1], wall, 1, 'wcs')
print(scrt)

sof=sc.skyoffset_frame()

# This image is only 2 dimensional. We need extra axes frequency and stokes.

m31image4D=image_replicate(m31image, shape=[1, 1, 4, len(frequency)])
m31sm = skymodel_from_image(m31image4D)


# In[ ]:

comp1= create_skycomponent(numpy.array([[1.0, 0.0, 0.0, 0.0]]), frequency)
m31sm=skymodel_add_component(m31sm, comp1)


# Now we can predict the visibility from this model

# In[ ]:

kwargs={'wstep':100.0, 'npixel':256, 'cellsize':0.0001}
vt = predict(vt, m31sm, **kwargs)


# To check that we got it right, plot the amplitude of the visibility.

# In[ ]:

uvdist=numpy.sqrt(vt.data['uvw'][:,0]**2+vt.data['uvw'][:,1]**2)
plt.clf()
plt.plot(uvdist, numpy.abs(vt.data['vis'][:,0,0]), '.')
plt.xlabel('uvdist')
plt.ylabel('Amp Visibility')
plt.show()


# Make the dirty image and point spread function

# In[ ]:

kwargs={}
kwargs['npixel']=512
kwargs['cellsize']=0.0001
kwargs['wstep']=30.0
dirty, psf, sumwt = invert(vt, **kwargs)
image_show(dirty)
print("Max, min in dirty image = %.6f, %.6f, sum of weights = %f" % (dirty.data.max(), dirty.data.min(), sumwt))

print("Max, min in PSF         = %.6f, %.6f, sum of weights = %f" % (psf.data.max(), psf.data.min(), sumwt))

image_to_fits(dirty, 'dirty.fits')
image_to_fits(psf, 'psf.fits')
m31compnew = fitcomponent(dirty, **kwargs)


# In[ ]:

kwargs={'niter':100, 'threshold':0.001, 'fracthresh':0.01}
comp, residual = clean(dirty, psf, **kwargs)


# In[ ]:




# In[ ]:

fig=image_show(comp)
fig=image_show(residual)


# In[ ]:

kwargs={'wstep':30.0}
vt = predict(vt, m31sm, **kwargs)
modelsm=skymodel_from_image(comp)
vtmodel = create_visibility(vlaa, times, frequency, weight=1.0, phasecentre=phasecentre)
vtmodel.data = vt.data.copy()
vtmodel=predict(vtmodel, modelsm,**kwargs)


# Now we will plot the original visibility and the residual visibility.

# In[ ]:

uvdist=numpy.sqrt(vt.data['uvw'][:,0]**2+vt.data['uvw'][:,1]**2)
plt.clf()
plt.plot(uvdist, numpy.abs(vt.data['vis'][:,0,0]), '.', color='b')
plt.plot(uvdist, numpy.abs(vt.data['vis'][:,0,0]-vtmodel.data['vis'][:,0,0]), '.', color='r')
plt.xlabel('uvdist')
plt.ylabel('Amp Visibility')
plt.show()


# In[ ]:



