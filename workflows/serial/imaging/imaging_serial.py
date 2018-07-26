"""Manages the imaging context. This take a string and returns a dictionary containing:
 * Predict function
 * Invert function
 * image_iterator function
 * vis_iterator function

"""

import logging

import numpy

from data_models.memory_data_models import Visibility, Image
from workflows.shared.imaging.imaging_shared import imaging_context

from processing_components.image.gather_scatter import image_scatter_facets
from processing_components.image.operations import create_empty_image_like
from processing_components.imaging.base import normalize_sumwt
from processing_components.visibility.base import copy_visibility, create_visibility_from_rows
from processing_components.visibility.coalesce import convert_blockvisibility_to_visibility, convert_visibility_to_blockvisibility

log = logging.getLogger(__name__)


def invert_serial(vis, im: Image, dopsf=False, normalize=True, context='2d', inner=None, vis_slices=1,
                  facets=1, overlap=0, taper=None, **kwargs):
    """ Invert using algorithm specified by context:

     * 2d: Two-dimensional transform
     * wstack: wstacking with either vis_slices or wstack (spacing between w planes) set
     * wprojection: w projection with wstep (spacing between w places) set, also kernel='wprojection'
     * timeslice: snapshot imaging with either vis_slices or timeslice set. timeslice='auto' does every time
     * facets: Faceted imaging with facets facets on each axis
     * facets_wprojection: facets AND wprojection
     * facets_wstack: facets AND wstacking
     * wprojection_wstack: wprojection and wstacking


    :param vis:
    :param im:
    :param dopsf: Make the psf instead of the dirty image (False)
    :param normalize: Normalize by the sum of weights (True)
    :param context: Imaging context e.g. '2d', 'timeslice', etc.
    :param inner: Inner loop 'vis'|'image'
    :param kwargs:
    :return: Image, sum of weights
    """
    c = imaging_context(context)
    vis_iter = c['vis_iterator']
    invert = c['invert']

    if not isinstance(vis, Visibility):
        svis = convert_blockvisibility_to_visibility(vis)
    else:
        svis = vis
    
    resultimage = create_empty_image_like(im)
    
    totalwt = None
    for rows in vis_iter(svis, vis_slices=vis_slices):
        if numpy.sum(rows):
            visslice = create_visibility_from_rows(svis, rows)
            sumwt = 0.0
            workimage = create_empty_image_like(im)
            for dpatch in image_scatter_facets(workimage, facets=facets, overlap=overlap, taper=taper):
                result, sumwt = invert(visslice, dpatch, dopsf, normalize=False, facets=facets,
                                       vis_slices=vis_slices, **kwargs)
                # Ensure that we fill in the elements of dpatch instead of creating a new numpy arrray
                dpatch.data[...] = result.data[...]
            # Assume that sumwt is the same for all patches
            if totalwt is None:
                totalwt = sumwt
            else:
                totalwt += sumwt
            resultimage.data += workimage.data
    
    assert totalwt is not None, "No valid data found for imaging"
    if normalize:
        resultimage = normalize_sumwt(resultimage, totalwt)
    
    return resultimage, totalwt


def predict_serial(vis, model: Image, context='2d', inner=None, vis_slices=1, facets=1, overlap=0, taper=None,
                   **kwargs) -> Visibility:
    """Predict visibilities using algorithm specified by context
    
     * 2d: Two-dimensional transform
     * wstack: wstacking with either vis_slices or wstack (spacing between w planes) set
     * wprojection: w projection with wstep (spacing between w places) set, also kernel='wprojection'
     * timeslice: snapshot imaging with either vis_slices or timeslice set. timeslice='auto' does every time
     * facets: Faceted imaging with facets facets on each axis
     * facets_wprojection: facets AND wprojection
     * facets_wstack: facets AND wstacking
     * wprojection_wstack: wprojection and wstacking

    
    :param vis:
    :param model: Model image, used to determine image characteristics
    :param context: Imaging context e.g. '2d', 'timeslice', etc.
    :param inner: Inner loop 'vis'|'image'
    :param kwargs:
    :return:


    """
    c = imaging_context(context)
    vis_iter = c['vis_iterator']
    predict = c['predict']
    if inner is None:
        inner = c['inner']
    
    if not isinstance(vis, Visibility):
        svis = convert_blockvisibility_to_visibility(vis)
    else:
        svis = vis
    
    result = copy_visibility(vis, zero=True)
    
    for rows in vis_iter(svis, vis_slices=vis_slices):
        if numpy.sum(rows):
            visslice = create_visibility_from_rows(svis, rows)
            visslice.data['vis'][...] = 0.0
            for dpatch in image_scatter_facets(model, facets=facets, overlap=overlap, taper=taper):
                result.data['vis'][...] = 0.0
                result = predict(visslice, dpatch, **kwargs)
                svis.data['vis'][rows] += result.data['vis']

    if not isinstance(vis, Visibility):
        svis = convert_visibility_to_blockvisibility(svis)

    return svis
