import os
import sys

import numpy

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy import constants as const
from astropy.wcs.utils import pixel_to_skycoord

from arl.data.polarisation import PolarisationFrame
from arl.pipelines.functions import rcal
from arl.skycomponent.operations import create_skycomponent
from arl.util.run_unittests import run_unittests
from arl.util.testing_support import create_named_configuration, create_blockvisibility_iterator

lowcore = create_named_configuration('LOWBD2-CORE')
times = numpy.linspace(-3.0, +3.0, 7) * numpy.pi / 12.0
frequency = numpy.linspace(1.0e8, 1.50e8, 3)
channel_bandwidth = numpy.array([5e7, 5e7, 5e7])

# Define the component and give it some polarisation and spectral behaviour\n",
f = numpy.array([100.0, 20.0, -10.0, 1.0])
flux = numpy.array([f, 0.8 * f, 0.6 * f])

phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
compdirection = SkyCoord(ra=17.0 * u.deg, dec=-36.5 * u.deg, frame='icrs', equinox='J2000')
comp = create_skycomponent(flux=flux, frequency=frequency, direction=compdirection)

ingest = create_blockvisibility_iterator(lowcore, times=times,
                                         frequency=frequency,
                                         channel_bandwidth=channel_bandwidth, phasecentre=phasecentre, 
                                         weight=1, polarisation_frame=PolarisationFrame('linear'), 
                                         integration_time=1.0, number_integrations=1,
                                         components=comp, phase_error=0.1, amplitude_error=0.01)

rcal_pipeline = rcal(vis=ingest, components=comp, phase_only=False)

for igt, gt in enumerate(rcal_pipeline):
    print (igt)

