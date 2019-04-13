"""Functions to help with persistence of data models

These do data conversion and persistence. Functions from processing_library and processing_components are used.
"""

import ast
import collections

import astropy.units as u
import h5py
import numpy
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.units import Quantity
from astropy.wcs import WCS

from processing_library.image.operations import create_image_from_array
from processing_components.griddata.operations import create_griddata_from_array
from processing_components.griddata.convolution_functions import create_convolutionfunction_from_array

from processing_components.image.operations import export_image_to_fits, import_image_from_fits
from data_models.memory_data_models import Visibility, BlockVisibility, Configuration, \
    GainTable, SkyModel, Skycomponent, Image, GridData, ConvolutionFunction, PointingTable
from data_models.polarisation import PolarisationFrame, ReceptorFrame


def convert_earthlocation_to_string(el: EarthLocation):
    """Convert Earth Location to string

    :param el:
    :return:
    """
    return "%s, %s, %s" % (el.x, el.y, el.z)


def convert_earthlocation_from_string(s: str):
    """Convert Earth Location to string

    :param s: String

    :return:
    """
    x, y, z = s.split(',')
    el = EarthLocation(x=Quantity(x), y=Quantity(y), z=Quantity(z))
    return el


def convert_direction_to_string(d: SkyCoord):
    """Convert SkyCoord to string

    TODO: Make more general!

    :param d: SkyCoord
    :return:
    """
    return "%s, %s, %s" % (d.ra.deg, d.dec.deg, 'icrs')


def convert_direction_from_string(s: str):
    """Convert direction (SkyCoord) from string
    
    TODO: Make more general!

    :param s: String

    :return:
    """
    ra, dec, frame = s.split(',')
    d = SkyCoord(ra, dec, unit='deg', frame=frame.strip())
    return d


def convert_configuration_to_hdf(config: Configuration, f):
    """

    :param config:
    :param f:
    :return:
    """
    cf = f.create_group('configuration')
    cf.attrs['ARL_data_model'] = 'Configuration'
    cf.attrs['name'] = config.name
    cf.attrs['location'] = convert_earthlocation_to_string(config.location)
    cf.attrs['frame'] = config.frame
    cf.attrs['receptor_frame'] = config.frame
    
    cf['configuration/xyz'] = config.xyz
    cf['configuration/diameter'] = config.diameter
    cf['configuration/names'] = [numpy.string_(name) for name in config.names]
    cf['configuration/mount'] = [numpy.string_(mount) for mount in config.mount]
    return f


def convert_configuration_from_hdf(f):
    """ Extyract configuration from HDF

    :param f:
    :return:
    """
    cf = f['configuration']
    
    assert cf.attrs['ARL_data_model'] == "Configuration", "%s is a Configuration" % cf.attrs['ARL_data_model']
    
    name = cf.attrs['name']
    location = convert_earthlocation_from_string(cf.attrs['location'])
    frame = cf.attrs['frame']
    receptor_frame = cf.attrs['receptor_frame']
    
    xyz = cf['configuration/xyz']
    diameter = cf['configuration/diameter']
    names = [str(n) for n in cf['configuration/names']]
    mount = [str(m) for m in cf['configuration/mount']]
    return Configuration(name=name, location=location, frame=frame, receptor_frame=receptor_frame, xyz=xyz,
                         diameter=diameter, names=names, mount=mount)


def convert_visibility_to_hdf(vis, f):
    """ Convert visibility to HDF

    :param vis:
    :param f: HDF root
    :return:
    """
    assert isinstance(vis, Visibility)
    
    f.attrs['ARL_data_model'] = 'Visibility'
    f.attrs['nvis'] = vis.nvis
    f.attrs['npol'] = vis.npol
    f.attrs['phasecentre_coords'] = vis.phasecentre.to_string()
    f.attrs['phasecentre_frame'] = vis.phasecentre.frame.name
    f.attrs['polarisation_frame'] = vis.polarisation_frame.type
    f['data'] = vis.data
    f = convert_configuration_to_hdf(vis.configuration, f)
    return f


def convert_hdf_to_visibility(f):
    """ Convert HDF root to visibility

    :param f:
    :return:
    """
    assert f.attrs['ARL_data_model'] == "Visibility", "Not a Visibility"
    s = f.attrs['phasecentre_coords'].split()
    ss = [float(s[0]), float(s[1])] * u.deg
    phasecentre = SkyCoord(ra=ss[0], dec=ss[1], frame=f.attrs['phasecentre_frame'])
    polarisation_frame = PolarisationFrame(f.attrs['polarisation_frame'])
    data = numpy.array(f['data'])
    vis = Visibility(data=data, polarisation_frame=polarisation_frame,
                     phasecentre=phasecentre)
    vis.configuration = convert_configuration_from_hdf(f)
    return vis


def convert_blockvisibility_to_hdf(vis: BlockVisibility, f):
    """ Convert blockvisibility to HDF

    :param vis:
    :param f: HDF root
    :return:
    """
    assert isinstance(vis, BlockVisibility)
    
    f.attrs['ARL_data_model'] = 'BlockVisibility'
    f.attrs['nvis'] = vis.nvis
    f.attrs['npol'] = vis.npol
    f.attrs['phasecentre_coords'] = vis.phasecentre.to_string()
    f.attrs['phasecentre_frame'] = vis.phasecentre.frame.name
    f.attrs['polarisation_frame'] = vis.polarisation_frame.type
    f.attrs['frequency'] = vis.frequency
    f.attrs['channel_bandwidth'] = vis.channel_bandwidth
    f['data'] = vis.data
    f = convert_configuration_to_hdf(vis.configuration, f)
    return f


def convert_hdf_to_blockvisibility(f):
    """ Convert HDF root to blockvisibility

    :param f:
    :return:
    """
    assert f.attrs['ARL_data_model'] == "BlockVisibility", "Not a BlockVisibility"
    s = f.attrs['phasecentre_coords'].split()
    ss = [float(s[0]), float(s[1])] * u.deg
    phasecentre = SkyCoord(ra=ss[0], dec=ss[1], frame=f.attrs['phasecentre_frame'])
    polarisation_frame = PolarisationFrame(f.attrs['polarisation_frame'])
    frequency = f.attrs['frequency']
    channel_bandwidth = f.attrs['channel_bandwidth']
    data = numpy.array(f['data'])
    vis = BlockVisibility(data=data, polarisation_frame=polarisation_frame,
                          phasecentre=phasecentre, frequency=frequency,
                          channel_bandwidth=channel_bandwidth)
    vis.configuration = convert_configuration_from_hdf(f)
    return vis


def export_visibility_to_hdf5(vis, filename):
    """ Export a Visibility to HDF5 format

    :param vis:
    :param filename:
    :return:
    """
    
    if not isinstance(vis, collections.Iterable):
        vis = [vis]
    with h5py.File(filename, 'w') as f:
        f.attrs['number_data_models'] = len(vis)
        for i, v in enumerate(vis):
            vf = f.create_group('Visibility%d' % i)
            convert_visibility_to_hdf(v, vf)
        f.flush()


def import_visibility_from_hdf5(filename):
    """Import a Visibility from HDF5 format

    :param filename:
    :return: If only one then a Visibility, otherwise a list of Visibilitys
    """
    
    with h5py.File(filename, 'r') as f:
        nvislist = f.attrs['number_data_models']
        vislist = [convert_hdf_to_visibility(f['Visibility%d' % i]) for i in range(nvislist)]
        if nvislist == 1:
            return vislist[0]
        else:
            return vislist


def export_blockvisibility_to_hdf5(vis, filename):
    """ Export a BlockVisibility to HDF5 format

    :param vis:
    :param filename:
    :return:
    """
    
    if not isinstance(vis, collections.Iterable):
        vis = [vis]
    with h5py.File(filename, 'w') as f:
        f.attrs['number_data_models'] = len(vis)
        for i, v in enumerate(vis):
            assert isinstance(v, BlockVisibility)
            vf = f.create_group('BlockVisibility%d' % i)
            convert_blockvisibility_to_hdf(v, vf)
        f.flush()


def import_blockvisibility_from_hdf5(filename):
    """Import a Visibility from HDF5 format

    :param filename:
    :return: If only one then a BlockVisibility, otherwise a list of BlockVisibility's
    """
    
    with h5py.File(filename, 'r') as f:
        nvislist = f.attrs['number_data_models']
        vislist = [convert_hdf_to_blockvisibility(f['BlockVisibility%d' % i]) for i in range(nvislist)]
        if nvislist == 1:
            return vislist[0]
        else:
            return vislist


def convert_gaintable_to_hdf(gt: GainTable, f):
    """ Convert GainTable to HDF

    :param gt:
    :param f: HDF root
    :return:
    """
    assert isinstance(gt, GainTable)
    
    f.attrs['ARL_data_model'] = 'GainTable'
    f.attrs['frequency'] = gt.frequency
    f.attrs['receptor_frame'] = gt.receptor_frame.type
    f.attrs['phasecentre_coords'] = gt.phasecentre.to_string()
    f.attrs['phasecentre_frame'] = gt.phasecentre.frame.name
    f['data'] = gt.data
    return f


def convert_hdf_to_gaintable(f):
    """ Convert HDF root to a GainTable

    :param f:
    :return:
    """
    assert f.attrs['ARL_data_model'] == "GainTable", "Not a GainTable"
    receptor_frame = ReceptorFrame(f.attrs['receptor_frame'])
    frequency = numpy.array(f.attrs['frequency'])
    data = numpy.array(f['data'])
    s = f.attrs['phasecentre_coords'].split()
    ss = [float(s[0]), float(s[1])] * u.deg
    phasecentre = SkyCoord(ra=ss[0], dec=ss[1], frame=f.attrs['phasecentre_frame'])
    gt = GainTable(data=data, receptor_frame=receptor_frame, frequency=frequency, phasecentre=phasecentre)
    return gt


def export_gaintable_to_hdf5(gt: GainTable, filename):
    """ Export a GainTable to HDF5 format

    :param gt:
    :param filename:
    :return:
    """
    
    if not isinstance(gt, collections.Iterable):
        gt = [gt]
    with h5py.File(filename, 'w') as f:
        f.attrs['number_data_models'] = len(gt)
        for i, g in enumerate(gt):
            assert isinstance(g, GainTable)
            gf = f.create_group('GainTable%d' % i)
            convert_gaintable_to_hdf(g, gf)
        f.flush()


def import_gaintable_from_hdf5(filename):
    """Import GainTable(s) from HDF5 format

    :param filename:
    :return: single gaintable or list of gaintables
    """
    
    with h5py.File(filename, 'r') as f:
        ngtlist = f.attrs['number_data_models']
        gtlist = [convert_hdf_to_gaintable(f['GainTable%d' % i]) for i in range(ngtlist)]
        if ngtlist == 1:
            return gtlist[0]
        else:
            return gtlist


def convert_pointingtable_to_hdf(pt: PointingTable, f):
    """ Convert PointingTable to HDF

    :param pt:
    :param f: HDF root
    :return:
    """
    assert isinstance(pt, PointingTable)
    
    f.attrs['ARL_data_model'] = 'PointingTable'
    f.attrs['frequency'] = pt.frequency
    f.attrs['receptor_frame'] = pt.receptor_frame.type
    f.attrs['pointingcentre_coords'] = pt.pointingcentre.to_string()
    f.attrs['pointingcentre_frame'] = pt.pointingcentre.frame.name
    f.attrs['pointing_frame'] = pt.pointing_frame
    f['data'] = pt.data
    return f


def convert_hdf_to_pointingtable(f):
    """ Convert HDF root to a PointingTable

    :param f:
    :return:
    """
    assert f.attrs['ARL_data_model'] == "PointingTable", "Not a PointingTable"
    receptor_frame = ReceptorFrame(f.attrs['receptor_frame'])
    frequency = numpy.array(f.attrs['frequency'])
    data = numpy.array(f['data'])
    s = f.attrs['pointingcentre_coords'].split()
    ss = [float(s[0]), float(s[1])] * u.deg
    pointingcentre = SkyCoord(ra=ss[0], dec=ss[1], frame=f.attrs['pointingcentre_frame'])
    pointing_frame = f.attrs['pointing_frame']
    pt = PointingTable(data=data, frequency=frequency, receptor_frame=receptor_frame, pointing_frame=pointing_frame,
                       pointingcentre=pointingcentre)
    return pt


def export_pointingtable_to_hdf5(pt: PointingTable, filename):
    """ Export a PointingTable to HDF5 format

    :param pt:
    :param filename:
    :return:
    """
    
    if not isinstance(pt, collections.Iterable):
        pt = [pt]
    with h5py.File(filename, 'w') as f:
        f.attrs['number_data_models'] = len(pt)
        for i, g in enumerate(pt):
            assert isinstance(g, PointingTable)
            gf = f.create_group('PointingTable%d' % i)
            convert_pointingtable_to_hdf(g, gf)
        f.flush()


def import_pointingtable_from_hdf5(filename):
    """Import PointingTable(s) from HDF5 format

    :param filename:
    :return: single pointingtable or list of pointingtables
    """
    
    with h5py.File(filename, 'r') as f:
        nptlist = f.attrs['number_data_models']
        ptlist = [convert_hdf_to_pointingtable(f['PointingTable%d' % i]) for i in range(nptlist)]
        if nptlist == 1:
            return ptlist[0]
        else:
            return ptlist


def convert_skycomponent_to_hdf(sc: Skycomponent, f):
    """ Convert Skycomponent to HDF
    :param sc: SkyComponent
    :param f: HDF root
    :return:
    """
    assert isinstance(sc, Skycomponent)
    
    f.attrs['ARL_data_model'] = 'Skycomponent'
    f.attrs['direction'] = convert_direction_to_string(sc.direction)
    f.attrs['frequency'] = sc.frequency
    f.attrs['polarisation_frame'] = sc.polarisation_frame.type
    f.attrs['flux'] = sc.flux
    f.attrs['shape'] = sc.shape
    f.attrs['params'] = str(sc.params)
    f.attrs['name'] = numpy.string_(sc.name)
    return f


def convert_hdf_to_skycomponent(f):
    """ Convert HDF root to a SkyComponent

    :param f:
    :return:
    """
    assert f.attrs['ARL_data_model'] == "Skycomponent", "Not a Skycomponent"
    direction = convert_direction_from_string(f.attrs['direction'])
    frequency = numpy.array(f.attrs['frequency'])
    name = f.attrs['name']
    polarisation_frame = PolarisationFrame(f.attrs['polarisation_frame'])
    flux = f.attrs['flux']
    shape = f.attrs['shape']
    params = ast.literal_eval(f.attrs['params'])
    sc = Skycomponent(direction=direction, frequency=frequency, name=name,
                      flux=flux, polarisation_frame=polarisation_frame,
                      shape=shape, params=params)
    return sc


def export_skycomponent_to_hdf5(sc: Skycomponent, filename):
    """ Export a Skycomponent to HDF5 format

    :param sc: SkyComponent
    :param filename:
    :return:
    """
    
    if not isinstance(sc, collections.Iterable):
        sc = [sc]
    with h5py.File(filename, 'w') as f:
        f.attrs['number_data_models'] = len(sc)
        for i, s in enumerate(sc):
            assert isinstance(s, Skycomponent)
            sf = f.create_group('Skycomponent%d' % i)
            convert_skycomponent_to_hdf(s, sf)
        f.flush()


def import_skycomponent_from_hdf5(filename):
    """Import Skycomponent(s) from HDF5 format

    :param filename:
    :return: single skycomponent or list of skycomponents
    """
    
    with h5py.File(filename, 'r') as f:
        nsclist = f.attrs['number_data_models']
        sclist = [convert_hdf_to_skycomponent(f['Skycomponent%d' % i]) for i in range(nsclist)]
        if nsclist == 1:
            return sclist[0]
        else:
            return sclist


def convert_image_to_hdf(im: Image, f):
    """ Convert Image to HDF

    :param im: Image
    :param f: HDF root
    :return:
    """
    if isinstance(im, Image):
    
        f.attrs['ARL_data_model'] = 'Image'
        f['data'] = im.data
        f.attrs['wcs'] = numpy.string_(im.wcs.to_header_string())
        f.attrs['polarisation_frame'] = im.polarisation_frame.type
        
    return f


def convert_hdf_to_image(f):
    """ Convert HDF root to an Image

    :param f:
    :return:
    """
    if 'ARL_data_model' in f.attrs.keys() and f.attrs['ARL_data_model'] == "Image":
        data = numpy.array(f['data'])
        polarisation_frame = PolarisationFrame(f.attrs['polarisation_frame'])
        wcs = WCS(f.attrs['wcs'])
        im = create_image_from_array(data, wcs=wcs,
                                 polarisation_frame=polarisation_frame)
        return im
    else:
        return None


def export_image_to_hdf5(im, filename):
    """ Export an Image to HDF5 format

    :param im:
    :param filename:
    :return:
    """
    
    if not isinstance(im, collections.Iterable):
        im = [im]
    with h5py.File(filename, 'w') as f:
        f.attrs['number_data_models'] = len(im)
        for i, m in enumerate(im):
            assert isinstance(m, Image)
            mf = f.create_group('Image%d' % i)
            convert_image_to_hdf(m, mf)
        f.flush()
        f.close()


def import_image_from_hdf5(filename):
    """Import Image(s) from HDF5 format

    :param filename:
    :return: single image or list of images
    """
    
    with h5py.File(filename, 'r') as f:
        nimlist = f.attrs['number_data_models']
        imlist = [convert_hdf_to_image(f['Image%d' % i]) for i in range(nimlist)]
        if nimlist == 1:
            return imlist[0]
        else:
            return imlist


def export_skymodel_to_hdf5(sm, filename):
    """ Export a Skymodel to HDF5 format

    :param sm:
    :param filename:
    :return:
    """
    
    if not isinstance(sm, collections.Iterable):
        sm = [sm]
    
    with h5py.File(filename, 'w') as f:
        f.attrs['number_data_models'] = len(sm)
        for i, s in enumerate(sm):
            assert isinstance(s, SkyModel)
            sf = f.create_group('SkyModel%d' % i)
            convert_skymodel_to_hdf(s, sf)
        f.flush()
        f.close()
        
def convert_skymodel_to_hdf(sm, f):
    """
    
    :param sm:
    :param f:
    :return:
    """
    f.attrs['ARL_data_model'] = 'SkyModel'
    f.attrs['fixed'] = sm.fixed
    if sm.components is not None:
        f.attrs['number_skycomponents'] = len(sm.components)
        for i, sc in enumerate(sm.components):
            cf = f.create_group('skycomponent%d' % i)
            convert_skycomponent_to_hdf(sm.components[i], cf)
        else:
            f.attrs['number_skycomponents'] = len(sm.components)
    if sm.image is not None:
        cf = f.create_group('image')
        convert_image_to_hdf(sm.image, cf)
    if sm.mask is not None:
        cf = f.create_group('mask')
        convert_image_to_hdf(sm.mask, cf)
    if sm.gaintable is not None:
        cf = f.create_group('gaintable')
        convert_gaintable_to_hdf(sm.gaintable, cf)
    return f


def import_skymodel_from_hdf5(filename):
    """Import a Skymodel from HDF5 format

    :param filename:
    :return: SkyModel
    """
    
    with h5py.File(filename, 'r') as f:
        nsmlist = f.attrs['number_data_models']
        smlist = [convert_hdf_to_skymodel(f['SkyModel%d' % i]) for i in range(nsmlist)]
        if nsmlist == 1:
            return smlist[0]
        else:
            return smlist
    
def convert_hdf_to_skymodel(f):
    """
    
    :param f:
    :return:
    """
    assert f.attrs['ARL_data_model'] == "SkyModel", f.attrs['ARL_data_model']

    fixed = f.attrs['fixed']

    ncomponents = f.attrs['number_skycomponents']
    components = list()
    for i in range(ncomponents):
        cf = f[('skycomponent%d' % i)]
        components.append(convert_hdf_to_skycomponent(cf))
    if 'image' in f.keys():
        cf = f['image']
        image = convert_hdf_to_image(cf)
    else:
        image = None
    if 'mask' in f.keys():
        cf = f['mask']
        mask = convert_hdf_to_image(cf)
    else:
        mask = None
    if 'gaintable' in f.keys():
        cf = f['gaintable']
        gaintable = convert_hdf_to_gaintable(cf)
    else:
        gaintable = None

    
    return SkyModel(image=image, components=components, gaintable=gaintable, mask=mask, fixed=fixed)


def convert_griddata_to_hdf(gd: GridData, f):
    """ Convert Griddata to HDF

    :param im: GridData
    :param f: HDF root
    :return:
    """
    assert isinstance(gd, GridData)
    
    f.attrs['ARL_data_model'] = 'GridData'
    f['data'] = gd.data
    f.attrs['grid_wcs'] = numpy.string_(gd.grid_wcs.to_header_string())
    f.attrs['projection_wcs'] = numpy.string_(gd.projection_wcs.to_header_string())
    f.attrs['polarisation_frame'] = gd.polarisation_frame.type
    return f


def convert_hdf_to_griddata(f):
    """ Convert HDF root to a GridData

    :param f:
    :return:
    """
    assert f.attrs['ARL_data_model'] == "GridData", "Not a GridData"
    data = numpy.array(f['data'])
    polarisation_frame = PolarisationFrame(f.attrs['polarisation_frame'])
    grid_wcs = WCS(f.attrs['grid_wcs'])
    projection_wcs = WCS(f.attrs['projection_wcs'])
    gd = create_griddata_from_array(data, grid_wcs=grid_wcs, projection_wcs=projection_wcs,
                                 polarisation_frame=polarisation_frame)
    return gd


def export_griddata_to_hdf5(gd, filename):
    """ Export a GridData to HDF5 format

    :param gd:
    :param filename:
    :return:
    """
    
    if not isinstance(gd, collections.Iterable):
        gd = [gd]
    with h5py.File(filename, 'w') as f:
        f.attrs['number_data_models'] = len(gd)
        for i, m in enumerate(gd):
            assert isinstance(m, GridData)
            mf = f.create_group('GridData%d' % i)
            convert_griddata_to_hdf(m, mf)
        f.flush()
        f.close()


def import_griddata_from_hdf5(filename):
    """Import GridData from HDF5 format

    :param filename:
    :return: single image or list of images
    """
    
    with h5py.File(filename, 'r') as f:
        nimlist = f.attrs['number_data_models']
        gdlist = [convert_hdf_to_griddata(f['GridData%d' % i]) for i in range(nimlist)]
        if nimlist == 1:
            return gdlist[0]
        else:
            return gdlist


def convert_convolutionfunction_to_hdf(cf: ConvolutionFunction, f):
    """ Convert Griddata to HDF

    :param im: ConvolutionFunction
    :param f: HDF root
    :return:
    """
    assert isinstance(cf, ConvolutionFunction)
    
    f.attrs['ARL_data_model'] = 'ConvolutionFunction'
    f['data'] = cf.data
    f.attrs['grid_wcs'] = numpy.string_(cf.grid_wcs.to_header_string())
    f.attrs['projection_wcs'] = numpy.string_(cf.projection_wcs.to_header_string())
    f.attrs['polarisation_frame'] = cf.polarisation_frame.type
    return f


def convert_hdf_to_convolutionfunction(f):
    """ Convert HDF root to a ConvolutionFunction

    :param f:
    :return:
    """
    assert f.attrs['ARL_data_model'] == "ConvolutionFunction", "Not a ConvolutionFunction"
    data = numpy.array(f['data'])
    polarisation_frame = PolarisationFrame(f.attrs['polarisation_frame'])
    grid_wcs = WCS(f.attrs['grid_wcs'])
    projection_wcs = WCS(f.attrs['projection_wcs'])
    gd = create_convolutionfunction_from_array(data, grid_wcs=grid_wcs, projection_wcs=projection_wcs,
                                    polarisation_frame=polarisation_frame)
    return gd


def export_convolutionfunction_to_hdf5(cf, filename):
    """ Export a ConvolutionFunction to HDF5 format

    :param cf:
    :param filename:
    :return:
    """
    
    if not isinstance(cf, collections.Iterable):
        cf = [cf]
    with h5py.File(filename, 'w') as f:
        f.attrs['number_data_models'] = len(cf)
        for i, m in enumerate(cf):
            assert isinstance(m, ConvolutionFunction)
            mf = f.create_group('ConvolutionFunction%d' % i)
            convert_convolutionfunction_to_hdf(m, mf)
        f.flush()
        f.close()


def import_convolutionfunction_from_hdf5(filename):
    """Import ConvolutionFunction from HDF5 format

    :param filename:
    :return: single image or list of images
    """
    
    with h5py.File(filename, 'r') as f:
        nimlist = f.attrs['number_data_models']
        cflist = [convert_hdf_to_convolutionfunction(f['ConvolutionFunction%d' % i]) for i in range(nimlist)]
        if nimlist == 1:
            return cflist[0]
        else:
            return cflist


def memory_data_model_to_buffer(model, jbuff, dm):
    """ Copy a memory data model to a buffer data model

    The file type is derived from the file extension. All are hdf only with the exception of Imaghe which can also be
    fits.

    :param model: Memory data model to be sent to buffer
    :param jbuff: JSON describing buffer
    :param dm: JSON describing data model
    """
    name = jbuff["directory"] + dm["name"]
    
    import os
    _, file_extension = os.path.splitext(dm["name"])
    
    if dm["data_model"] == "BlockVisibility":
        return export_blockvisibility_to_hdf5(model, name)
    elif dm["data_model"] == "Image":
        if file_extension == ".fits":
            return export_image_to_fits(model, name)
        else:
            return export_image_to_hdf5(model, name)
    elif dm["data_model"] == "GridData":
        return export_griddata_to_hdf5(model, name)
    elif dm["data_model"] == "ConvolutionFunction":
        return export_convolutionfunction_to_hdf5(model, name)
    elif dm["data_model"] == "SkyModel":
        return export_skymodel_to_hdf5(model, name)
    elif dm["data_model"] == "GainTable":
        return export_gaintable_to_hdf5(model, name)
    elif dm["data_model"] == "PointingTable":
        return export_pointingtable_to_hdf5(model, name)
    else:
        raise ValueError("Data model %s not supported" % dm["data_model"])


def buffer_data_model_to_memory(jbuff, dm):
    """Copy a buffer data model into memory data model

    The file type is derived from the file extension. All are hdf only with the exception of Imaghe which can also be
    fits.

    :param jbuff: JSON describing buffer
    :param dm: JSON describing data model
    :return: data model
    """
    import os
    name = os.path.join(jbuff["directory"], dm["name"])
    
    import os
    _, file_extension = os.path.splitext(dm["name"])
    
    if dm["data_model"] == "BlockVisibility":
        return import_blockvisibility_from_hdf5(name)
    elif dm["data_model"] == "Image":
        if file_extension == ".fits":
            return import_image_from_fits(name)
        else:
            return import_image_from_hdf5(name)
    elif dm["data_model"] == "SkyModel":
        return import_skymodel_from_hdf5(name)
    elif dm["data_model"] == "GainTable":
        return import_gaintable_from_hdf5(name)
    elif dm["data_model"] == "PointingTable":
        return import_pointingtable_from_hdf5(name)
    elif dm["data_model"] == "GridData":
        return import_griddata_from_hdf5(name)
    elif dm["data_model"] == "ConvolutionFunction":
        return import_convolutionfunction_from_hdf5(name)
    else:
        raise ValueError("Data model %s not supported" % dm["data_model"])


