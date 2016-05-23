# -*- coding: utf-8 -*-
"""Simple CASA imaging script.

Run with:
    casapy --nogui --nologger --log2term -c casa_image.py <ms>

See:
    http://casa.nrao.edu/docs/CasaRef/CasaRef.html
for documentation of methods on CASA objects 'im' and 'ia'
"""

import os
import time
import sys
import shutil
import math


def fov_to_cell_size(fov, im_size):
    """Convert field-of-view in degrees and number of pixels to cell-size.

    Args:
        fov (float) : Field-of-view, in degrees.
        im_size (int) : Image size in pixels.

    Returns:
        Image pixel size (cell size), in arcsec.
    """
    r_max = math.sin(math.radians(fov) / 2.)
    inc = r_max / (0.5 * im_size)
    return math.degrees(math.asin(inc)) * 3600.


# -------------------------------------
ms_path = os.path.abspath(sys.argv[-1])
ms_name = os.path.basename(ms_path)
image_root_name = os.path.splitext(ms_path)[0]
size = 1024
fov = 3.0  # deg
# fov = 0.2  # deg
cell = fov_to_cell_size(fov, size)  # arcsec
im_size = [size, size]
cell_size = ['%.10farcsec' % cell, '%.10farcsec' % cell]
w_planes = 0
make_psf = False
grid_function = 'SF'  # SF | BOX
new_phase_centre = False
ra0 = -90.3545848760
dec0 = -11.1711239906
# -------------------------------------

if not os.path.isdir(ms_path):
    raise RuntimeError('Specified MS not found!')

im.open(ms_path, usescratch=False, compress=False)
im.defineimage(nx=im_size[0], ny=im_size[1],
               cellx=cell_size[0], celly=cell_size[1],
               stokes='I', mode='mfs', step=1, spw=[-1], outframe='',
               veltype='radio')
if new_phase_centre:
    im.defineimage(nx=im_size[0], ny=im_size[1],
                   cellx=cell_size[0], celly=cell_size[1],
                   stokes='I', mode='mfs', step=1, spw=[-1], outframe='',
                   veltype='radio',
                   phasecenter=me.direction('J2000', '%.14fdeg' % ra0,
                                            '%.14fdeg' % dec0))

im.weight(type='natural')
if w_planes > 0:
    im.setoptions(ftmachine='wproject', wprojplanes=w_planes,
                  gridfunction=grid_function,
                  padding=1.2, dopbgriddingcorrections=True,
                  applypointingoffsets=False)
else:
    im.setoptions(ftmachine='ft', gridfunction=grid_function,
                  padding=1.0, dopbgriddingcorrections=True,
                  applypointingoffsets=False)
dirty_image = image_root_name + '_dirty_s%04i_f%04.1f_w%03i' % (size, fov,
                                                                w_planes)
t0 = time.time()
print '*' * 80
print '* Starting imaging...'
im.makeimage(image=dirty_image + '.img', type='observed', verbose=True)
print '* Time taken to make dirty image = %.3f s' % (time.time() - t0)
print '*' * 80
if make_psf:
    psf_image = image_root_name + '_psf_s%04i_f%04.1f_w%03i' % (size, fov,
                                                                w_planes)
    im.makeimage(image=psf_image + '.img', type='psf', verbose=True)
im.close()
ia.open(dirty_image + '.img')
ia.tofits(dirty_image + '.fits', overwrite=True)
ia.close()
shutil.rmtree(dirty_image + '.img')

if make_psf:
    ia.open(psf_image + '.img')
    ia.tofits(psf_image + '.fits', overwrite=True)
    ia.close()
    shutil.rmtree(psf_image + '.img')
