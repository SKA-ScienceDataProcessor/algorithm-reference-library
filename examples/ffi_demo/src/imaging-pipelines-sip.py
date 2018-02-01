
# coding: utf-8

# # Pipeline processing
# 
# This notebook demonstrates the continuum imaging and ICAL pipelines.

import os
import sys
sys.path.append(os.path.join('..','..'))

results_dir = './results'
os.makedirs(results_dir, exist_ok=True)

import numpy

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs.utils import pixel_to_skycoord

from matplotlib import pyplot as plt

from arl.calibration.solvers import solve_gaintable
from arl.calibration.operations import apply_gaintable, create_gaintable_from_blockvisibility
from arl.data.data_models import Image
from arl.data.polarisation import PolarisationFrame
from arl.data.parameters import get_parameter
from arl.visibility.base import create_blockvisibility
from arl.skycomponent.operations import create_skycomponent
from arl.image.operations import show_image, export_image_to_fits, qa_image, copy_image, create_empty_image_like
from arl.visibility.iterators import vis_timeslice_iter
from arl.visibility.coalesce import convert_visibility_to_blockvisibility
from arl.util.testing_support import create_named_configuration, create_low_test_beam,     create_low_test_image_from_gleam, simulate_gaintable
from arl.imaging import create_image_from_visibility, advise_wide_field
from arl.imaging.imaging_context import invert_function, predict_function
from arl.pipelines.functions import ical

import logging

log = logging.getLogger()
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(sys.stdout))


# We create a graph to make the visibility. The parameter rmax determines the distance of the furthest antenna/stations used. All over parameters are determined from this number.

nfreqwin=5
ntimes=11
rmax=300.0
frequency=numpy.linspace(0.8e8,1.2e8,nfreqwin)
channel_bandwidth=numpy.array(nfreqwin*[frequency[1]-frequency[0]])
times = numpy.linspace(-numpy.pi/3.0, numpy.pi/3.0, ntimes)
phasecentre=SkyCoord(ra=+30.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')

lowcore = create_named_configuration('LOWBD2-CORE', rmax=rmax)

block_vis = create_blockvisibility(lowcore, times, frequency=frequency, channel_bandwidth=channel_bandwidth,
                     weight=1.0, phasecentre=phasecentre,
                     polarisation_frame=PolarisationFrame("stokesI"))


wprojection_planes=1
advice=advise_wide_field(block_vis, guard_band_image=4.0, delA=0.02,
                             wprojection_planes=wprojection_planes)

vis_slices = advice['vis_slices']
npixel=advice['npixels2']
cellsize=advice['cellsize']


gleam_model = create_low_test_image_from_gleam(npixel=npixel, frequency=frequency,
    channel_bandwidth=channel_bandwidth, cellsize=cellsize, phasecentre=phasecentre, applybeam=True)


predicted_vis = predict_function(block_vis, gleam_model, vis_slices=51, context='wstack')
block_vis=convert_visibility_to_blockvisibility(predicted_vis)
gt = create_gaintable_from_blockvisibility(block_vis)
gt = simulate_gaintable(gt, phase_error=1.0)
blockvis = apply_gaintable(block_vis, gt)


model = create_image_from_visibility(block_vis, npixel=npixel, frequency=[numpy.average(frequency)], nchan=1,
    channel_bandwidth=[numpy.sum(channel_bandwidth)], cellsize=cellsize, phasecentre=phasecentre)


dirty, sumwt = invert_function(predicted_vis, model, vis_slices=vis_slices, dopsf=False, context='wstack')


deconvolved, residual, restored = ical(block_vis=blockvis, model=model, vis_slices=vis_slices, timeslice='auto',
                                                  algorithm='hogbom', niter=1000, fractional_threshold=0.1, threshold=0.1,
                                                  context='wstack', nmajor=5, gain=0.1, first_selfcal=1,
                                                  global_solution=False)
print(qa_image(deconvolved, context='Clean image'))
export_image_to_fits(deconvolved, '%s/imaging-dask_ical_deconvolved.fits'
                     % (results_dir))

print(qa_image(residual, context='Residual clean image'))
export_image_to_fits(residual, '%s/imaging-dask_ical_residual.fits'
                     % (results_dir))

print(qa_image(restored, context='Restored clean image'))
export_image_to_fits(restored, '%s/imaging-dask_ical_restored.fits'
                     % (results_dir))

