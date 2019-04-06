""" Functions for ionospheric modelling: see SDP memo 97

"""

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord

from data_models.memory_data_models import BlockVisibility
from processing_components.calibration.operations import create_gaintable_from_blockvisibility, \
    create_gaintable_from_rows
from processing_components.calibration.iterators import gaintable_timeslice_iter
from processing_components.image.operations import copy_image, create_empty_image_like
from processing_components.visibility.base import create_visibility_from_rows
from processing_components.visibility.iterators import vis_timeslice_iter
from processing_library.util.coordinate_support import xyz_to_uvw, skycoord_to_lmn

import logging
log = logging.getLogger(__name__)

def create_gaintable_from_pointingtable(vis, sc, pt, vp, vis_slices=None, scale=1.0, **kwargs):
    """ Create gaintables from a pointing table

    :param vis:
    :param sc: Sky components for which pierce points are needed
    :param pt: Pointing table
    :param vp: Voltage pattern
    :param scale: Multiply the screen by this factor
    :return:
    """
    assert isinstance(vis, BlockVisibility)
    
    nant = vis.vis.shape[1]
    gaintables = [create_gaintable_from_blockvisibility(vis, **kwargs) for i in sc]
    
    # The time in the Visibility is hour angle in seconds!
    for iha, rows in enumerate(vis_timeslice_iter(vis, vis_slices=vis_slices)):
        v = create_visibility_from_rows(vis, rows)
        ha = numpy.average(v.time)
        pt_rows = (pt.time == ha)
        pointing_ha = pt.pointing[pt_rows]
        number_bad = 0
        number_good = 0

        r2d = 180.0 / numpy.pi
        for icomp, comp in enumerate(sc):
            antgain = numpy.zeros([nant], dtype='complex')
            for ant in range(nant):
                worldloc = [float((comp.direction.ra.rad + pointing_ha[0, ant, 0, 0, 0])*r2d) ,
                            float((comp.direction.dec.rad + pointing_ha[0, ant, 0, 0, 1])*r2d),
                            1.0, 1e9]
                try:
                    pixloc = vp.wcs.wcs_world2pix([worldloc], 0)[0].astype('int')
                    antgain[ant] = 1.0 / (scale * vp.data[pixloc[3], pixloc[2], pixloc[1], pixloc[0]])
                    number_good += 1
                except:
                    number_bad += 1
                    antgain[ant] = 0.0
            
            gaintables[icomp].gain[iha, :, :, :] = antgain[:, numpy.newaxis, numpy.newaxis, numpy.newaxis]
            gaintables[icomp].phasecentre = comp.direction
        
        if number_bad > 0:
            log.warning("create_gaintable_from_pointingtable: %d points are inside the voltage pattern image" % (number_good))
            log.warning("create_gaintable_from_pointingtable: %d points are outside the voltage pattern image" % (number_bad))

    return gaintables