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

from functions.clean import clean
from functions.visibility import create_visibility
from functions.imaging import *
from functions.skymodel import SkyModel, skymodel_from_image, skymodel_add_image, skymodel_add_component
from functions.skycomponent import *
from functions.image import image_show, image_from_fits, image_to_fits, image_replicate, fitcomponent
from functions.configuration import configuration_filter, named_configuration

kwargs = {}

vlaa = configuration_filter(named_configuration('VLAA'), **kwargs)
vlaa.data['xyz']=vlaa.data['xyz']/10.0

times = numpy.arange(-numpy.pi/2.0, +numpy.pi/2.0,0.05)
frequency = numpy.array([1e8])

reffrequency = numpy.max(frequency)
phasecentre = SkyCoord(0.0*u.rad, u.rad*numpy.pi/4, frame='icrs', equinox=2000.0)
vt = create_visibility(vlaa, times, frequency, weight=1.0, phasecentre=phasecentre)

plt.clf()
for f in frequency:
    x=f/const.c
    plt.plot(x*vt.data['uvw'][:,0], x*vt.data['uvw'][:,1], '.', color='b')
    plt.plot(-x*vt.data['uvw'][:,0], -x*vt.data['uvw'][:,1], '.', color='r')