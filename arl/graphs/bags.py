""" Calibration and imaging functions converted to Dask.bags. `Dask <http://dask.pydata.org/>`_ is a
python-based flexible parallel computing library for analytic computing.

Bags uses a filter-map-reduce approach. If the operations are on bag of similar objects (e.g. all Visibility's) then
just the map function is needed with perhaps a suitable adpater function is needed. See for example safe_invert_list
and sake_predict_list.

If different bags are to tbe combined, they can be zipped together before being passed to an adapter function. As
a result the bag related functions below are quite thin and it may be better to go directly to the underlayed bag
filter-map-reduce functions.

In this interface, we use a dictionary interface to provide some structure of the contents of the bags. The bag has
consists of a list of dictionary items each of which meta data such as freqwin, timewin, etc. For examples,
a record used in imaging or calibration would be::

    [   {   'freqwin': 0,
            'vis': <arl.data.data_models.Visibility object at 0x11486e710>},
        {   'freqwin': 1,
            'vis': <arl.data.data_models.Visibility object at 0x11486e6d8>},
        {   'freqwin': 2,
            'vis': <arl.data.data_models.Visibility object at 0x11486ef98>},
        {   'freqwin': 3,
            'vis': <arl.data.data_models.Visibility object at 0x11486ecc0>},
        {   'freqwin': 4,
            'vis': <arl.data.data_models.Visibility object at 0x11486e828>},
        {   'freqwin': 5,
            'vis': <arl.data.data_models.Visibility object at 0x105ced438>},
        {   'freqwin': 6,
            'vis': <arl.data.data_models.Visibility object at 0x105ced320>}]

This then is scattered in w to give a large number of records such as::

    {'freqwin': 0, 'vis': <arl.data.data_models.Visibility object at 0x11a1d40b8>, 'wstack': 42}

Functions called invert_record and predict_record are mapped across this and the results reduced using the
dask.bag.foldby function::

    dirty_bag = vis_bag\
        .map(invert_record, model, dopsf=False, context='wstack')\
        .foldby('freqwin', binop=invert_binop, initial=initial_record)\
        .map(folded_to_record, 'freqwin')

There are some helper functions invert_binop, folded_to_record that do conversions as necessary.

The output from invert is then a list of records::

    [   {   'freqwin': 0,
            'image': (   <arl.data.data_models.Image object at 0x12ba12e80>,
                         array([[ 0.06530285]]))},
        {   'freqwin': 1,
            'image': (   <arl.data.data_models.Image object at 0x12ba12f28>,
                         array([[ 0.06530285]]))},
        {   'freqwin': 2,
            'image': (   <arl.data.data_models.Image object at 0x111ae5588>,
                         array([[ 0.06530285]]))},
        {   'freqwin': 3,
            'image': (   <arl.data.data_models.Image object at 0x12bc34fd0>,
                         array([[ 0.06530285]]))},
        {   'freqwin': 4,
            'image': (   <arl.data.data_models.Image object at 0x112d4fc18>,
                         array([[ 0.06530285]]))},
        {   'freqwin': 5,
            'image': (   <arl.data.data_models.Image object at 0x111afe8d0>,
                         array([[ 0.06530285]]))},
        {   'freqwin': 6,
            'image': (   <arl.data.data_models.Image object at 0x112d4f3c8>,
                         array([[ 0.06530285]]))}]

Note that all parameters here should be passed using the kwargs mechanism.
"""
import logging

import numpy
from dask import bag

from arl.calibration.operations import qa_gaintable, apply_gaintable
from arl.calibration.solvers import solve_gaintable
from arl.data.data_models import Image
from arl.image.deconvolution import deconvolve_cube, restore_cube
from arl.image.operations import create_image_from_array, qa_image, create_empty_image_like
from arl.imaging import normalize_sumwt
from arl.imaging.imaging_context import imaging_context, predict_context, invert_context
from arl.visibility.base import copy_visibility, create_visibility_from_rows
from arl.visibility.coalesce import convert_visibility_to_blockvisibility
from arl.visibility.gather_scatter import visibility_gather_channel
from arl.visibility.operations import qa_visibility, sort_visibility, \
    divide_visibility, integrate_visibility_by_channel, subtract_visibility

log = logging.getLogger(__name__)


def reify(bg, compute=False):
    """Compute a bag and create a new bag to hold the contexts
    
    This is useful to avoid recalculating results when not necessary. It's also often necessary when a reduction of
    one component of a bag is required.
    
    :param bg:
    :param compute: .compute() instead of list()
    :return:
    """
    if isinstance(bg, bag.Bag):
        if compute:
            return bag.from_sequence(bg.compute())
        else:
            return bag.from_sequence(list(bg))
    else:
        return bg


def print_element(x, context='', indent=4, width=160):
    from pprint import PrettyPrinter
    pp = PrettyPrinter(indent=indent, width=width)
    if context == '':
        pp.pprint(x)
    else:
        pp.pprint("%s: %s" % (context, x))
    return x


def scatter_record(record, context, **kwargs):
    """ Scatter a record according to the context's vis_iter field.

    :param record:
    :param context: Imaging context
    :param kwargs:
    :return:
    """
    c = imaging_context(context)
    assert c['vis_iterator'] is not None, "Scatter not possible for context %s" % context
    vis_iter = c['vis_iterator']
    result = list()
    scatter_index = 0
    for rows in vis_iter(record['vis'], **kwargs):
        if rows is not None:
            v = create_visibility_from_rows(record['vis'], rows)
            newrecord = {}
            for key in record.keys():
                newrecord[key] = record[key]
            newrecord['vis'] = v
            newrecord[context] = scatter_index
            scatter_index += 1
            result.append(newrecord)
    return result


def join_records(r1, r2):
    """ Output is r1 except for overrides by r2

    :param r1
    :param r2:
    :return:
    """
    ro = r1
    for k2 in r2.keys():
        ro[k2] = r2[k2]
    return ro


def predict_record(record, context, **kwargs):
    """ Do a predict for a given record

    :param record:
    :param context:
    :param kwargs:
    :return:
    """
    newrecord = {}
    for key in record.keys():
        if key not in ['image', context]:
            newrecord[key] = record[key]
    if record['vis'] is not None:
        newvis = copy_visibility(record['vis'], zero=True)
        newrecord['vis'] = predict_context(newvis, record['image'], context=context, **kwargs)
    else:
        newrecord['vis'] = None
    return newrecord


def invert_record(record, dopsf, context, **kwargs):
    """ Do an invert for a given record

    :param record:
    :param dopsf:
    :param context:
    :param kwargs:
    :return:
    """
    assert 'vis' in record.keys(), "vis not contained in record keys %s" % record
    vis = record['vis']
    
    newrecord = {}
    for key in record.keys():
        if key != 'vis':
            newrecord[key] = record[key]
    if vis is not None:
        newrecord['image'] = invert_context(vis, record['image'], dopsf, context=context, **kwargs)
    else:
        newrecord['image'] = None
    return newrecord


def image_to_records_bag(nfreqwin, im):
    """ Wrap an image in records

    :param nfreqwin:
    :param im:
    :return:
    """
    
    def create(freqwin):
        return {'image': im,
                'freqwin': freqwin}
    
    # Return a bag to hold all the requests
    return bag.range(nfreqwin, npartitions=nfreqwin).map(create)


def map_record(record, apply_function, key='vis', **kwargs):
    """ Apply a function to a record

    :param record:
    :param apply_function: unary function to apply
    :param key: key in record
    :param kwargs:
    :return:
    """
    assert isinstance(key, str), "Key is not a string: %s" % key
    assert key in record.keys(), "%s not contained in record keys %s" % (key, record)
    rec = record[key]
    
    newrecord = {}
    for k in record.keys():
        if k != key:
            newrecord[k] = record[k]
        else:
            newrecord[k] = apply_function(rec, **kwargs)
    
    return newrecord


def invert_record_add(r1, r2, normalize=True):
    """ Add two invert records together

    :param r1:
    :param r2:
    :param normalize:
    :return:
    """
    im1 = r1['image']
    if im1 is None:
        return r2
    im2 = r2['image']
    if im2 is None:
        return r1
    
    result = create_image_from_array(im1[0].data * im1[1] + im2[0].data * im2[1],
                                     im1[0].wcs, im1[0].polarisation_frame)
    weight = im1[1] + im2[1]
    if normalize:
        result = normalize_sumwt(result, weight)
    
    newrecord = {}
    for key in r1.keys():
        if key != 'vis':
            newrecord[key] = r1[key]
    for key in r2.keys():
        if key != 'vis':
            newrecord[key] = r2[key]
    newrecord['image'] = (result, weight)
    return newrecord


def predict_record_concatenate(r1, r2):
    """ Add two predict records together

    :param r1:
    :param r2:
    :return:
    """
    vis1 = r1['vis']
    vis2 = r2['vis']
    if vis1 is None:
        return r2
    if vis2 is None:
        return r1
    assert vis1.polarisation_frame == vis2.polarisation_frame
    assert vis1.phasecentre.separation(vis2.phasecentre).value < 1e-15
    ovis = copy_visibility(vis1)
    ovis.data = numpy.hstack((vis1.data, vis2.data))
    newrecord = {}
    for key in r1.keys():
        if key != 'vis':
            newrecord[key] = r1[key]
    for key in r2.keys():
        if key != 'vis':
            newrecord[key] = r2[key]
    newrecord['vis'] = ovis
    return newrecord


def predict_record_subtract(r1, r2):
    """ Subtract two predict records

    :param r1:
    :param r2:
    :return:
    """
    vis1 = r1['vis']
    vis2 = r2['vis']
    assert vis1.polarisation_frame == vis2.polarisation_frame
    assert vis1.phasecentre.separation(vis2.phasecentre).value < 1e-15
    ovis = copy_visibility(vis1)
    ovis.data['vis'] -= vis2.vis
    newrecord = {}
    for key in r1.keys():
        newrecord[key] = r1[key]
    newrecord['vis'] = ovis
    return newrecord


def create_empty_image_record(model):
    """ Create an empty image record to be used in invert_record_add

    :param model:
    :return:
    """
    assert isinstance(model, Image)
    return {'image': (create_empty_image_like(model), 0.0)}


def create_empty_visibility_record():
    """ Create an empty visibility record to be used in predict_record_concatenate

    :return:
    """
    return {'vis': None}


def folded_to_image_record(folded):
    """ Convert the output from foldby back into our record format

    :param folded:
    :return:
    """
    return folded[1]


def folded_to_visibility_record(folded):
    """ Convert the output from foldby back into our record format

    :param folded:
    :return:
    """
    result = folded[1]
    result['vis'] = sort_visibility(folded[1]['vis'])
    return result


def invert_bag(vis_bag, model_bag, dopsf=False, context='2d', key='freqwin', **kwargs) -> bag:
    """ Construct a bag to invert a bag of visibilities to a bag of (image, weight) tuples

    Call directly - don't use via bag.map

    :param vis_bag:
    :param model_bag: This is just used as specification of the output images
    :param dopsf:
    :param context:
    :param kwargs:
    :return:
    """
    assert isinstance(vis_bag, bag.Bag), vis_bag
    assert isinstance(model_bag, bag.Bag), model_bag
    
    return vis_bag \
        .map(join_records, model_bag) \
        .map(scatter_record, context, **kwargs) \
        .flatten() \
        .map(invert_record, dopsf=dopsf, context=context, **kwargs) \
        .foldby(key, binop=invert_record_add) \
        .map(folded_to_image_record)


def predict_bag(vis_bag, model_bag, context='2d', key='freqwin', **kwargs) -> bag:
    """Construct a bag to predict a bag of visibilities.

    The vis_bag is scattered appropriately, the predict is applied, and the data then
    concatenated and sorted back to the original order from creation

    Call directly - don't use via bag.map

    :param vis_bag:
    :param model_bag: This must be a bag of images
    :param context:
    :param key:
    :param kwargs:
    :return:
    """
    # The steps here are:
    #
    # - Scatter the visibilities according to context
    # - Flatten to e.g. {'freqwin': 0, 'vis': <arl.data.data_models.Visibility object at 0x114a10208>, 'timeslice': 4}
    # - Do a predict on all records yielding e.g. {'freqwin': 1, 'vis': <arl.data.data_models.Visibility object at
    # 0x114a10a58>, 'timeslice': 3}
    # - Concatenate the results according to the key e.g. (1, {'freqwin': 1, 'timeslice': 4,
    # 'vis': <arl.data.data_models.Visibility object at 0x114a10da0>})
    # - Convert the results back to record structure
    #
    # Monitoring is via print_element
    assert isinstance(vis_bag, bag.Bag), vis_bag
    assert isinstance(model_bag, bag.Bag), model_bag
    
    return vis_bag \
        .map(join_records, model_bag) \
        .map(scatter_record, context=context, **kwargs) \
        .flatten() \
        .map(predict_record, context, **kwargs) \
        .foldby(key, binop=predict_record_concatenate) \
        .map(folded_to_visibility_record)


def deconvolve_bag(dirty_bag, psf_bag, model_bag, **kwargs) -> bag:
    """ Deconvolve a bag of images to obtain a bag of models

    Call directly - don't use via bag.map

    :param dirty_bag:
    :param psf_bag:
    :param model_bag:
    :param kwargs:
    :return: Bag of Images
    """
    
    assert isinstance(dirty_bag, bag.Bag), dirty_bag
    assert isinstance(psf_bag, bag.Bag), psf_bag
    assert isinstance(model_bag, bag.Bag), model_bag
    
    def deconvolve(dirty, psf, model):
        # The dirty and psf are actually (Image, weight) tuples.
        result = deconvolve_cube(dirty['image'][0], psf['image'][0], **kwargs)
        result[0].data += model['image'].data
        return {'image': result[0]}
    
    model_bag = reify(model_bag)
    dirty_bag = reify(dirty_bag)
    return dirty_bag \
        .map(deconvolve, psf_bag, model_bag)


def restore_bag(comp_bag, psf_bag, res_bag, **kwargs) -> bag:
    """ Restore a bag of images to obtain a bag of restored images

    Call directly - don't use via bag.map

    :param dirty_bag:
    :param psf_bag:
    :param res_bag:
    :param kwargs:
    :return: Bag of Images
    """
    assert isinstance(comp_bag, bag.Bag), comp_bag
    assert isinstance(psf_bag, bag.Bag), psf_bag
    assert isinstance(res_bag, bag.Bag), res_bag
    
    def restore(comp, psf, res):
        return restore_cube(comp['image'], psf['image'][0], res['image'][0], **kwargs)
    
    return comp_bag.map(restore, psf_bag, res_bag)


def residual_image_bag(vis_bag, model_image_bag, **kwargs) -> bag:
    """Calculate residual images

    Call directly - don't use via bag.map

    :param vis_bag: Bag containing visibilities
    :param model_image_bag: Model images, one per visibility in vis_bag
    :param kwargs:
    :return:
    """
    assert isinstance(vis_bag, bag.Bag), vis_bag
    assert isinstance(model_image_bag, bag.Bag), model_image_bag
    
    result_vis_bag = reify(predict_bag(vis_bag, model_image_bag, **kwargs))
    result_vis_bag = reify(vis_bag).map(predict_record_subtract, r2=result_vis_bag)
    return invert_bag(result_vis_bag, model_image_bag, dopsf=False, **kwargs)


def selfcal_bag(vis_bag, model_bag, **kwargs):
    """ Create a bag for (optionally global) selfcalibration of a list of visibilities

    If global solution is true then visibilities are gathered to a single visibility data set which is then
    self-calibrated. The resulting gaintable is then effectively scattered out for application to each visibility
    set. If global solution is false then the solutions are performed locally.

    :param vis_bag: Bag of observed visibilities
    :param model_bag: Bag of model visibilities
    :param vis_slices:
    :param global_solution: Solve for global gains?
    :param kwargs: Parameters for functions in bags
    :return:
    """
    assert isinstance(vis_bag, bag.Bag), vis_bag
    assert isinstance(model_bag, bag.Bag), model_bag
    
    vis_bag = reify(vis_bag)
    model_bag = reify(model_bag)
    model_vis_bag = vis_bag \
        .map(map_record, copy_visibility, zero=True)
    model_vis_bag = predict_bag(model_vis_bag, model_bag, **kwargs) \
        .map(map_record, convert_visibility_to_blockvisibility)
    return calibrate_bag(vis_bag, model_vis_bag, **kwargs)


def residual_vis_bag(vis_bag, model_vis_bag):
    """ Create a bag for subtraction of list of visibilities

    :param vis_bag: Bag of observed visibilities
    :param model_vis_bag: Bag of model visibilities
    :return:
    """
    assert isinstance(vis_bag, bag.Bag), vis_bag
    assert isinstance(model_vis_bag, bag.Bag), model_vis_bag
    
    def subtract(vis, modelvis):
        return subtract_visibility(vis, modelvis['vis'])
    
    vis_bag = reify(vis_bag)
    model_vis_bag = reify(model_vis_bag)
    return vis_bag.map(map_record, subtract, modelvis=model_vis_bag)


def calibrate_bag(vis_bag, model_vis_bag, global_solution=False, **kwargs):
    """ Create a bag for (optionally global) calibration of a list of visibilities

    If global solution is true then visibilities are gathered to a single visibility
    data set which is then self-calibrated. The resulting gaintable is then effectively
    scattered out for application to each visibility set. If global solution is false
    then the solutions are performed locally.

    :param vis_bag:
    :param model_vis_bag:
    :param global_solution: Solve for global gains
    :param kwargs: Parameters for functions in graphs
    :return:
    """
    
    assert isinstance(vis_bag, bag.Bag), vis_bag
    assert isinstance(model_vis_bag, bag.Bag), model_vis_bag
    
    if global_solution:
        def divide(vis, modelvis):
            return divide_visibility(vis, modelvis['vis'])
        
        model_vis_bag = reify(model_vis_bag)
        point_vis_bag = vis_bag \
            .map(map_record, divide, modelvis=model_vis_bag) \
            .map(map_record, visibility_gather_channel, **kwargs) \
            .map(map_record, integrate_visibility_by_channel, **kwargs)
        
        # This is a global solution so we only get one gain table
        gt_bag = point_vis_bag.map(map_record, solve_gaintable, modelvis=None, **kwargs)
        return vis_bag.map(map_record, apply_gaintable, gt=gt_bag, inverse=True, **kwargs)
    else:
        def solve_and_apply(vis, modelvis):
            gt = solve_gaintable(vis, modelvis['vis'], **kwargs)
            log.debug(qa_gaintable(gt, context='calibrate_bag'))
            return apply_gaintable(vis, gt, inverse=True)
        
        model_vis_bag = reify(model_vis_bag)
        return vis_bag \
            .map(map_record, solve_and_apply, modelvis=model_vis_bag)


def qa_visibility_bag(vis, context=''):
    """ Print qa on the visibilities, use this in a sequence of bag operations

    Can be used in bag.map() as a passthru

    :param vis:
    :return:
    """
    s = qa_visibility(vis, context=context)
    log.info(s)
    return vis


def qa_image_bag(im, context=''):
    """ Print qa on images, use this in a sequence of bag operations

    Can be used in bag.map() as a passthru

    :param im:
    :return:
    """
    s = qa_image(im, context=context)
    log.info(s)
    return im


def qa_gaintable_bag(gt, context=''):
    """ Print qa on gaintables, use this in a sequence of bag operations

    Can be used in bag.map() as a passthru

    :param gt:
    :return:
    """
    s = qa_gaintable(gt, context=context)
    log.info(s)
    return gt
