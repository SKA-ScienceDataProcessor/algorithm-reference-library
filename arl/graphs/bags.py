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
from arl.image.deconvolution import deconvolve_cube, restore_cube
from arl.image.operations import create_image_from_array, qa_image, create_empty_image_like
from arl.imaging import normalize_sumwt
from arl.imaging.imaging_context import imaging_context
from arl.visibility.base import copy_visibility
from arl.visibility.gather_scatter import visibility_gather_channel
from arl.visibility.operations import qa_visibility, sort_visibility, \
    divide_visibility, integrate_visibility_by_channel
from arl.visibility.coalesce import convert_visibility_to_blockvisibility, \
    convert_blockvisibility_to_visibility


def reify(bg):
    if isinstance(bg, bag.Bag):
        return bag.from_sequence(bg.compute())
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


log = logging.getLogger(__name__)


def scatter_record(record, model, context, **kwargs):
    """ Scatter a record according to the context's scatter field.

    :param record:
    :param context: Imaging context
    :param kwargs:
    :return:
    """
    log.debug("Into scatter_record", context, record)
    c = imaging_context(context)
    assert c['scatter'] is not None, "Scatter not possible for context %s" % context
    scatter = c['scatter']
    result = list()
    vis_list = scatter(record['vis'], **kwargs)
    scatter_index = 0
    for v in vis_list:
        if v is not None:
            newrecord = {}
            for key in record.keys():
                newrecord[key] = record[key]
            newrecord['vis'] = v
            newrecord[context] = scatter_index
            scatter_index += 1
            newrecord['model']=model
            result.append(newrecord)
    log.debug("From scatter_record", result)
    return result


def predict_record(record, context, **kwargs):
    """ Do a predict for a given record

    :param record:
    :param model:
    :param context:
    :param kwargs:
    :return:
    """
    c = imaging_context(context)
    predict = c['predict']
    newrecord = {}
    for key in record.keys():
        newrecord[key] = record[key]
    newvis = copy_visibility(record['vis'], zero=True)
    newrecord['vis'] = predict(newvis, record['model'], context=context, **kwargs)
    return newrecord


def invert_record(record, dopsf, context, **kwargs):
    """ Do an invert for a given record

    :param record:
    :param model:
    :param dopsf:
    :param context:
    :param kwargs:
    :return:
    """
    c = imaging_context(context)
    invert = c['invert']
    
    assert 'vis' in record.keys(), "vis not contained in record keys %s" % record
    vis = record['vis']
    
    newrecord = {}
    for key in record.keys():
        if key != 'vis':
            newrecord[key] = record[key]
    newrecord['image'] = invert(vis, record['model'], dopsf, **kwargs)
    return newrecord


def map_record(record, apply_function, key='vis', **kwargs):
    """ Apply a function to a record

    :param record:
    :param apply_function: unary function to apply
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
    im2 = r2['image']
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
    :param normalize:
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
    :param normalize:
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
    return {'image': (create_empty_image_like(model), 0.0)}


def create_empty_visibility_record(vis):
    """ Create an empty visibility record to be used in predict_record_concatenate

    :param model:
    :return:
    """
    return {'vis': None}


def folded_to_image_record(folded):
    """ Convert the output from foldby back into our record format

    :param folded:
    :param key:
    :return:
    """
    return folded[1]


def folded_to_visibility_record(folded):
    """ Convert the output from foldby back into our record format

    :param folded:
    :param key:
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
    :param context:
    :param kwargs:
    :return:
    """
    return vis_bag \
        .map(scatter_record, model_bag, context, **kwargs) \
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
    return vis_bag \
        .map(scatter_record, model_bag, context=context, **kwargs) \
        .flatten() \
        .map(predict_record, context, **kwargs) \
        .foldby(key, binop=predict_record_concatenate) \
        .map(folded_to_visibility_record)


def deconvolve_bag(dirty_bag, psf_bag, model_bag, **kwargs) -> bag:
    """ Deconvolve a bag of images to obtain a bag of models

    Call directly - don't use via bag.map

    :param dirty_bag:
    :param psf_bag:
    :param kwargs:
    :return: Bag of Images
    """
    
    def deconvolve(dirty, psf, model, **kwargs):
        # The dirty and psf are actually (Image, weight) tuples.
        result = deconvolve_cube(dirty['image'][0], psf['image'][0], **kwargs)
        result[0].data += model.data
        return result[0]
    
    return dirty_bag \
        .map(deconvolve, psf_bag, model_bag, **kwargs)


def restore_bag(comp_bag, psf_bag, res_bag, **kwargs) -> bag:
    """ Restore a bag of images to obtain a bag of restored images

    Call directly - don't use via bag.map

    :param dirty_bag:
    :param psf_bag:
    :param kwargs:
    :return: Bag of Images
    """
    
    def restore(comp, psf, res, **kwargs):
        return restore_cube(comp, psf['image'][0], res['image'][0])
    
    return comp_bag.map(restore, psf_bag, res_bag, **kwargs)


def residual_image_bag(vis_bag, model_image_bag, context='2d', **kwargs) -> bag:
    """Calculate residual images

    Call directly - don't use via bag.map

    :param vis_bag: Bag containing visibilities
    :param model_image_bag: Model images, one per visibility in vis_bag
    :param kwargs:
    :return:
    """
    result_vis_bag = reify(predict_bag(vis_bag, model_image_bag, context=context, **kwargs))
    result_vis_bag = reify(vis_bag).map(predict_record_subtract, result_vis_bag)
    return invert_bag(result_vis_bag, model_image_bag, context=context, **kwargs)


def selfcal_bag(vis_bag, model_bag, **kwargs):
    """ Create a bag for (optionally global) selfcalibration of a list of visibilities

    If global solution is true then visibilities are gathered to a single visibility data set which is then
    self-calibrated. The resulting gaintable is then effectively scattered out for application to each visibility
    set. If global solution is false then the solutions are performed locally.

    :param vis_bag: Bag of observed visibilities
    :param model_bag: Bag of model visibilities
    :param vis_slices:
    :param global_solution: Solve for global gains?
    :param kwargs: Parameters for functions in graphs
    :return:
    """
    vis_bag = reify(vis_bag)
    model_bag = reify(model_bag)
    model_vis_bag = vis_bag\
        .map(map_record, copy_visibility, zero=True)
    model_vis_bag = predict_bag(model_vis_bag, model_bag, **kwargs)\
        .map(map_record, convert_visibility_to_blockvisibility)
    return calibrate_bag(vis_bag, model_vis_bag, **kwargs)


def calibrate_bag(vis_bag, model_vis_bag, global_solution=True, **kwargs):
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
    
    if global_solution:
        def divide(vis, modelvis):
            return divide_visibility(vis, modelvis['vis'])
        
        model_vis_bag = reify(model_vis_bag)
        point_vis_bag = vis_bag\
            .map(map_record, divide, modelvis=model_vis_bag) \
            .map(map_record, visibility_gather_channel, **kwargs) \
            .map(map_record, integrate_visibility_by_channel, **kwargs)
        
        # This is a global solution so we only get one gain table
        gt_bag = point_vis_bag.map(map_record, solve_gaintable, modelvis=None, **kwargs)
        return vis_bag.map(map_record, apply_gaintable, gt=gt_bag, inverse=True, **kwargs)
    else:
        def solve_and_apply(vis, modelvis, **kwargs):
            gt = solve_gaintable(vis, modelvis['vis'], **kwargs)
            return apply_gaintable(vis, gt, inverse=True)
        model_vis_bag = reify(model_vis_bag)
        return vis_bag.map(map_record, solve_and_apply, modelvis=model_vis_bag, **kwargs)


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
