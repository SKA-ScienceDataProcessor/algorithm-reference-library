"Functions to help with persistence of data models"

import ast

import astropy.units as u
import h5py
import numpy
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.units import Quantity
from astropy.wcs import WCS

from libs.image.operations import create_image_from_array
from .memory_data_models import Visibility, BlockVisibility, Configuration, \
    GainTable, SkyModel, Skycomponent, Image
from .polarisation import PolarisationFrame, ReceptorFrame


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
    """Convert Direction to string

    TODO: Make more general!

    :param el:
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
    """

    :param config:
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
    f.close()
    return vis


def convert_blockvisibility_to_hdf(vis: BlockVisibility, f):
    """ Convert blockvisibility to HDF

    :param vis:
    :param f: HDF root
    :return:
    """
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
    f.close()
    return vis


def export_visibility_to_hdf5(vis, filename):
    """ Export a Visibility to HDF5 format

    :param vis:
    :param filename:
    :return:
    """
    
    assert isinstance(vis, Visibility)
    with h5py.File(filename, 'w') as f:
        convert_visibility_to_hdf(vis, f)
        f.flush()
        f.close()


def import_visibility_from_hdf5(filename):
    """Import a Visibility from HDF5 format

    :param filename:
    :return:
    """
    
    with h5py.File(filename, 'r') as f:
        return convert_hdf_to_visibility(f)


def export_blockvisibility_to_hdf5(vis: BlockVisibility, filename):
    """ Export a Visibility to HDF5 format

    :param vis:
    :param filename:
    :return:
    """
    
    assert isinstance(vis, BlockVisibility)
    with h5py.File(filename, 'w') as f:
        convert_blockvisibility_to_hdf(vis, f)
        f.flush()
        f.close()


def import_blockvisibility_from_hdf5(filename):
    """Import a BlockVisibility from HDF5 format

    :param filename:
    :return:
    """
    
    with h5py.File(filename, 'r') as f:
        return convert_hdf_to_blockvisibility(f)


def convert_gaintable_to_hdf(gt: GainTable, f):
    """ Convert GainTable to HDF

    :param gt:
    :param f: HDF root
    :return:
    """
    f.attrs['ARL_data_model'] = 'GainTable'
    f.attrs['frequency'] = gt.frequency
    f.attrs['receptor_frame'] = gt.receptor_frame.type
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
    gt = GainTable(data=data, receptor_frame=receptor_frame, frequency=frequency)
    f.close()
    return gt


def export_gaintable_to_hdf5(gt: GainTable, filename):
    """ Export a GainTable to HDF5 format

    :param gt:
    :param filename:
    :return:
    """
    
    assert isinstance(gt, GainTable)
    with h5py.File(filename, 'w') as f:
        convert_gaintable_to_hdf(gt, f)
        f.flush()
        f.close()


def import_gaintable_from_hdf5(filename):
    """Import a GainTabley from HDF5 format

    :param filename:
    :return:
    """
    
    with h5py.File(filename, 'r') as f:
        return convert_hdf_to_gaintable(f)


def convert_skycomponent_to_hdf(sc: Skycomponent, f):
    """ Convert Skycomponent to HDF

    :param gt:
    :param f: HDF root
    :return:
    """
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
    """ Convert HDF root to a GainTable

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


def export_skycomponent_to_hdf5(sc, filename):
    """ Export a Skycomponent to HDF5 format

    :param gt:
    :param filename:
    :return:
    """
    
    assert isinstance(sc, Skycomponent)
    with h5py.File(filename, 'w') as f:
        convert_skycomponent_to_hdf(sc, f)
        f.flush()
        f.close()


def import_skycomponent_from_hdf5(filename):
    """Import a Skycomponent from HDF5 format

    :param filename:
    :return:
    """
    
    with h5py.File(filename, 'r') as f:
        return convert_hdf_to_skycomponent(f)


def convert_image_to_hdf(im: Image, f):
    """ Convert Image to HDF

    :param gt:
    :param f: HDF root
    :return:
    """
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
    assert f.attrs['ARL_data_model'] == "Image", "Not an Image"
    data = numpy.array(f['data'])
    polarisation_frame = PolarisationFrame(f.attrs['polarisation_frame'])
    wcs = WCS(f.attrs['wcs'])
    im = create_image_from_array(data, wcs=wcs,
                                 polarisation_frame=polarisation_frame)
    return im


def export_image_to_hdf5(im, filename):
    """ Export an Image to HDF5 format

    :param gt:
    :param filename:
    :return:
    """
    
    assert isinstance(im, Image)
    with h5py.File(filename, 'w') as f:
        convert_image_to_hdf(im, f)
        f.flush()
        f.close()


def import_image_from_hdf5(filename):
    """Import an Image from HDF5 format

    :param filename:
    :return:
    """
    
    with h5py.File(filename, 'r') as f:
        return convert_hdf_to_image(f)


def export_skymodel_to_hdf5(sm, filename):
    """ Export a Skymodel to HDF5 format

    :param sm:
    :param filename:
    :return:
    """
    
    assert isinstance(sm, SkyModel)
    with h5py.File(filename, 'w') as f:
        f.attrs['number_skycomponents'] = len(sm.components)
        for i, sc in enumerate(sm.components):
            cf = f.create_group('skycomponent%d' % i)
            convert_skycomponent_to_hdf(sc, cf)
        f.attrs['number_images'] = len(sm.images)
        for i, im in enumerate(sm.images):
            cf = f.create_group('image%d' % i)
            convert_image_to_hdf(im, cf)
        
        f.flush()
        f.close()


def import_skymodel_from_hdf5(filename):
    """Import a Skymodel from HDF5 format

    :param filename:
    :return: SkyModel
    """
    
    with h5py.File(filename, 'r') as f:
        ncomponents = f.attrs['number_skycomponents']
        components = [convert_hdf_to_skycomponent(f['skycomponent%d' % i])
                      for i in range(ncomponents)]
        
        nimages = f.attrs['number_images']
        images = [convert_hdf_to_image(f['image%d' % i]) for i in range(nimages)]

        return SkyModel(components=components, images=images)