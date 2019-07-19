#!/usr/bin/env python
# coding: utf-8

# # Simulation of the effect of pointing errors on MID observations

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import sys

from data_models.parameters import arl_path

results_dir = arl_path('test_results')

import numpy

from astropy.coordinates import SkyCoord
from astropy import units as u

from matplotlib import pyplot as plt

from data_models.polarisation import PolarisationFrame
from data_models.memory_data_models import Skycomponent, SkyModel

from wrappers.serial.visibility.base import create_blockvisibility
from wrappers.serial.image.operations import show_image, qa_image
from wrappers.serial.simulation.testing_support import create_named_configuration, simulate_pointingtable
from wrappers.serial.imaging.primary_beams import create_vp, create_pb
from wrappers.serial.skycomponent.operations import apply_beam_to_skycomponent
from wrappers.serial.imaging.base import create_image_from_visibility, advise_wide_field
from processing_components.calibration.pointing import create_pointingtable_from_blockvisibility
from processing_components.simulation.pointing import create_gaintable_from_pointingtable
from wrappers.arlexecute.visibility.base import copy_visibility

from wrappers.arlexecute.visibility.coalesce import convert_blockvisibility_to_visibility

from workflows.arlexecute.skymodel.skymodel_arlexecute import predict_skymodel_list_arlexecute_workflow
from workflows.arlexecute.imaging.imaging_arlexecute import invert_list_arlexecute_workflow
from workflows.serial.imaging.imaging_serial import weight_list_serial_workflow

from wrappers.arlexecute.execution_support.arlexecute import arlexecute

from wrappers.arlexecute.execution_support.dask_init import get_dask_Client


import logging
log = logging.getLogger()
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))
mpl_logger = logging.getLogger("matplotlib") 
mpl_logger.setLevel(logging.WARNING) 


# In[3]:


from matplotlib import pylab

pylab.rcParams['figure.figsize'] = (12.0, 12.0)
pylab.rcParams['image.cmap'] = 'rainbow'
doplot=False


# In[4]:


client = get_dask_Client(memory_limit=8 * 1024 * 1024 * 1024)
arlexecute.set_client(client=client)


# In[5]:


nfreqwin = 1
ntimes = 65
rmax = 1e5
diameter=15.0

frequency = [1.4e9]
channel_bandwidth = [1e7]

HWHM_deg = 180.0*3e8/(numpy.pi*diameter*frequency[0])

print('HWHM beam = %g deg' % HWHM_deg)
HWHM = HWHM_deg * numpy.pi / 180.0

h2r = numpy.pi / 12.0
times = numpy.linspace(-6 * h2r, +6 * h2r, ntimes)

phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-45.0 * u.deg, frame='icrs', equinox='J2000')
midcore = create_named_configuration('MID', rmax=rmax)

block_vis = create_blockvisibility(midcore, times, frequency=frequency,
                                   channel_bandwidth=channel_bandwidth, weight=1.0, phasecentre=phasecentre,
                                   polarisation_frame=PolarisationFrame("stokesI"), zerow=True)


# In[6]:


vis = convert_blockvisibility_to_visibility(block_vis)
advice=advise_wide_field(vis, guard_band_image=1.0, delA=0.02)

cellsize=advice['cellsize']
npixel=512
pb_npixel=4096
pb_cellsize=1.03*HWHM/pb_npixel


# In[7]:


plt.clf()
plt.plot(vis.configuration.xyz[:,0], vis.configuration.xyz[:,1], '.')
plt.show()


# In[8]:


plt.clf()
plt.plot(vis.u, vis.v, '.')
plt.show()


# In[9]:


model = create_image_from_visibility(block_vis, npixel=npixel, frequency=frequency,
                                    nchan=nfreqwin, cellsize=cellsize, phasecentre=phasecentre)

vis = weight_list_serial_workflow([vis], [model])[0]


# ### Calculate the voltage patterns with and without pointing errors

# In[10]:


original_component = [Skycomponent(flux=[[1.0]], direction=phasecentre, 
                                  frequency=frequency, polarisation_frame=PolarisationFrame('stokesI'))]

vp = create_image_from_visibility(block_vis, npixel=pb_npixel, frequency=frequency,
                                    nchan=nfreqwin, cellsize=pb_cellsize, phasecentre=phasecentre,
                                 override_cellsize=False)

offset = [180.0*pb_cellsize*pb_npixel/(2.0*numpy.pi), 0.0]
HWHM=HWHM_deg * numpy.pi / 180.0
pb_direction = SkyCoord(ra=(+15.0 + offset[0]/numpy.cos(-45.0 * numpy.pi/180.0)) * u.deg, 
                        dec=(-45.0 + offset[1]) * u.deg, frame='icrs', equinox='J2000')
pb = create_pb(vp, 'MID', pointingcentre=pb_direction)
show_image(pb)
plt.show()
print(pb.data[0,0,pb_npixel//2,pb_npixel//2])
print(pb.data[0,0,0,pb_npixel//2])
print(pb.data[0,0,:,0])


# In[11]:


vp = create_vp(vp, 'MID', pointingcentre=pb_direction)
pt = create_pointingtable_from_blockvisibility(block_vis, vp)

no_error_pt = simulate_pointingtable(pt, 0.0, 0.0, seed=18051955)
no_error_gt = create_gaintable_from_pointingtable(block_vis, original_component, no_error_pt, vp)


# In[12]:


no_error_sm=SkyModel(components=original_component, gaintable=no_error_gt[0])
no_error_vis = copy_visibility(vis)
no_error_vis = predict_skymodel_list_arlexecute_workflow(no_error_vis, [no_error_sm], context='2d', docal=True)
no_error_vis=arlexecute.compute(no_error_vis, sync=True)[0]


# In[13]:


static=0.0
dynamic=1.0
fwhm = 1.0
pes = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]
qas = []
for pe in pes:
    static_pointing_error = static * pe * numpy.pi/(3600.0*180.0)
    pointing_error = dynamic * pe * numpy.pi/(3600.0*180.0)
    error_pt = simulate_pointingtable(pt, pointing_error=pointing_error, 
                                      static_pointing_error=static_pointing_error, seed=18051955)
    error_gt = create_gaintable_from_pointingtable(block_vis, original_component, error_pt, vp)

    error_sm=SkyModel(components=original_component, gaintable=error_gt[0])

    error_vis = copy_visibility(vis)
    error_vis = predict_skymodel_list_arlexecute_workflow(error_vis, [error_sm], context='2d', docal=True)
    error_vis=arlexecute.compute(error_vis, sync=True)[0]

    error_vis.data['vis']-=no_error_vis.data['vis']
    
    dirty = invert_list_arlexecute_workflow([error_vis], [model], '2d')
    dirty, sumwt = arlexecute.compute(dirty, sync=True)[0]
    qa=qa_image(dirty)
    print(qa)
    qas.append(qa)
    if True:
        show_image(dirty, cm='gray_r')
        plt.show()


# In[14]:


plt.clf()
for field in ['maxabs', 'rms', 'medianabs']:
    plt.loglog(pes, [q.data[field] for q in qas], '-', label=field)
plt.xlabel('Pointing error (arcsec)')
plt.ylabel('Error (Jy)')
if HWHM_deg > 0.0:
    plt.title('Error for 1Jy point source %g deg off axis at %g Hz, %d times, full array: dynamic %g, static %g' % 
              (HWHM_deg, frequency[0], ntimes, dynamic, static))

else:
    plt.title('Error for 1Jy point source on axis at %g Hz, %d times, full array: dynamic %g, static %g' % 
              (frequency[0], ntimes, dynamic, static))

    
plt.legend()
plt.show()


# In[ ]:




