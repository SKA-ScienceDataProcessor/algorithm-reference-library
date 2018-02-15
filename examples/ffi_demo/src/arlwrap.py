# Author: Bojan Nikolic <b.nikolic@mrao.cam.ac.uk>
# ARL Wrapping

import cffi
import numpy
import collections
import sys

from astropy.coordinates import SkyCoord
from astropy import units as u

from arl.calibration.operations import apply_gaintable, create_gaintable_from_blockvisibility
from arl.visibility.base import create_visibility, copy_visibility
from arl.data.data_models import Image, Visibility, BlockVisibility, ReceptorFrame, GainTable
from arl.image.deconvolution import deconvolve_cube, restore_cube
from arl.imaging.base import create_image_from_visibility, predict_2d
from arl.imaging import invert_2d, advise_wide_field
from arl.util.testing_support import create_named_configuration, create_test_image, create_low_test_image_from_gleam, simulate_gaintable
from arl.data.polarisation import PolarisationFrame
from arl.visibility.base import create_blockvisibility
from arl.imaging.imaging_context import predict_function 
from arl.visibility.coalesce import convert_visibility_to_blockvisibility


import pickle

ff = cffi.FFI()

ff.cdef("""
typedef struct {
  size_t nvis;
  int npol;
  void *data;
  char *phasecentre;
} ARLVis;
""")

def ARLDataVisSize(nvis, npol):
    return (80+32*int(npol))*int(nvis)

def cARLVis(visin):
    """
    Convert a const ARLVis * into the ARL Visiblity structure
    """
    npol=visin.npol
    nvis=visin.nvis
    #print (ARLDataVisSize(nvis, npol))
    desc = [('index', '>i8'),
            ('uvw', '>f8', (3,)),
            ('time', '>f8'),
            ('frequency', '>f8'),
            ('channel_bandwidth', '>f8'),
            ('integration_time', '>f8'),
            ('antenna1', '>i8'),
            ('antenna2', '>i8'),
            ('vis', '>c16', (npol,)),
            ('weight', '>f8', (npol,)),
            ('imaging_weight', '>f8', (npol,))]
    r=numpy.frombuffer(ff.buffer(visin.data,
                                 ARLDataVisSize(nvis, npol)),
                                 dtype=desc,
                                 count=nvis)
    return r


def ARLBlockDataVisSize(ntimes, nants, nchan, npol):
    return (24+24*int(nants*nants) + 24*int(nants*nants)*int(nchan)*int(npol))*int(ntimes)

def cARLBlockVis(visin, nants, nchan):
    """
    Convert a const ARLVis * into the ARL BlockVisiblity structure
    """
    npol=visin.npol
    ntimes=visin.nvis
    #print (ARLDataVisSize(nvis, npol))
    desc = [('index', '>i8'),
            ('uvw', '>f8', (nants, nants, 3)),
            ('time', '>f8'),
            ('integration_time', '>f8'),
            ('vis', '>c16', (nants, nants, nchan, npol)),
            ('weight', '>f8', (nants, nants, nchan, npol))]
    r=numpy.frombuffer(ff.buffer(visin.data,
                                 ARLBlockDataVisSize(ntimes, nants, nchan, npol)),
                                 dtype=desc,
                                 count=ntimes)
    return r


ff.cdef("""
typedef struct {
  size_t nrows;
  void *data;
} ARLGt;
""")

def ARLDataGTSize(ntimes, nants, nchan, nrec):
    return (8 + 8*nchan*nrec*nrec + 3*8*nants*nchan*nrec*nrec)*ntimes

def cARLGt(gtin, nants, nchan, nrec):
    """
    Convert a const ARLGt * into the ARL GainTable structure 
    """
    ntimes=gtin.nrows
    desc = [('gain', '>c16', (nants, nchan, nrec, nrec)),
            ('weight', '>f8', (nants, nchan, nrec, nrec)),
            ('residual', '>f8', (nchan, nrec, nrec)),
            ('time', '>f8')]
    r=numpy.frombuffer(ff.buffer(gtin.data,
                                 ARLDataGTSize(ntimes, nants, nchan, nrec)),
                                 dtype=desc,
                                 count=ntimes)
    return r


@ff.callback("void (*)(const ARLVis *, ARLVis *, bool)")
def arl_copy_visibility_ffi(visin, visout, zero):
    """
    Wrap of arl.visibility.base.copy_visibility
    """
    # Extra comments becasue this is an example.
    #
    # Convert the input visibilities into the ARL structure
    nvisin=cARLVis(visin)

    # Call the ARL function 
    tvis=copy_visibility(nvisin, zero=zero)

    # Copy the result into the output buffer
    visout.npol=visin.npol
    visout.nvis=visin.nvis
    nvisout=cARLVis(visout)
    numpy.copyto(nvisout, tvis)


arl_copy_visibility=collections.namedtuple("FFIX", "address")    
arl_copy_visibility.address=int(ff.cast("size_t", arl_copy_visibility_ffi))    


ff.cdef("""
typedef struct {
    size_t size;
    int data_shape[4];
    void *data;
    char *wcs;
    char *polarisation_frame;
} Image;
""")

def cImage(image_in, new=False):
    "Convert an Image* into ARL Image structure"
    new_image = Image()
    size = image_in.size
    data_shape = tuple(image_in.data_shape)
    new_image.data = numpy.frombuffer(ff.buffer(image_in.data,size*8),
            dtype='f8',
            count=size)

    # frombuffer only does 1D arrays..
    new_image.data = new_image.data.reshape(data_shape)

    # New images don't have pickles yet
    if new:
        new_image.wcs = numpy.frombuffer(ff.buffer(image_in.wcs,
            2996),
            dtype='b',
            count=2996)
        new_image.polarisation_frame = numpy.frombuffer(ff.buffer(
            image_in.polarisation_frame, 114),
            dtype='b',
            count=114)
    else:
        new_image.wcs = pickle.loads(ff.buffer(image_in.wcs, 2996))
        new_image.polarisation_frame = pickle.loads(ff.buffer(image_in.polarisation_frame,114))
    
    return new_image

def store_pickle(c_dest, py_src, raw_c_ptr=False):
    src_pickle = pickle.dumps(py_src)
    src_buf = numpy.frombuffer(src_pickle, dtype='b',
        count=len(src_pickle))

    # Create ndarray if necessary
    if (raw_c_ptr):
        c_dest = numpy.frombuffer(ff.buffer(c_dest, len(src_pickle)),
            dtype='b', count=len(src_pickle))

    numpy.copyto(c_dest, src_buf)

def load_pickle(c_ptr, size):
    return pickle.loads(numpy.frombuffer(ff.buffer(c_ptr, size),
        dtype='b',
        count=size))

def store_image_pickles(c_img, py_img):
    store_pickle(c_img.wcs, py_img.wcs)
    store_pickle(c_img.polarisation_frame, py_img.polarisation_frame)

# Turns ARLVis struct into Visibility object
def helper_create_visibility_object(c_vis):
    # This may be incorrect
    # especially the data field...
    tvis= Visibility(
            data=c_vis,
            frequency=c_vis['frequency'],
            channel_bandwidth=c_vis['channel_bandwidth'],
            integration_time=c_vis['integration_time'],
            antenna1=c_vis['antenna1'],
            antenna2=c_vis['antenna2'],
            weight=c_vis['weight'],
            imaging_weight=c_vis['imaging_weight'],
            uvw=c_vis['uvw'],
            time=c_vis['time']
            )
    return tvis

# Turns ARLVis struct into BlockVisibility object
def helper_create_blockvisibility_object(c_vis, freqs, chan_b, config):
    # This may be incorrect
    # especially the data field...
    tvis= BlockVisibility(
            data=c_vis,
    #        frequency=c_vis['frequency'],
    #        channel_bandwidth=c_vis['channel_bandwidth'],
	    frequency = freqs,
            channel_bandwidth = chan_b,
            configuration = config,
            integration_time=c_vis['integration_time'],
            weight=c_vis['weight'],
            uvw=c_vis['uvw'],
            time=c_vis['time']
            )
    return tvis

# Turns ARLGt struct into GainTable object
def helper_create_gaintable_object(c_gt, freqs):
    tgt= GainTable(
            data=c_gt,
	    frequency = freqs,
            gain=c_gt['gain'],
            weight=c_gt['weight'],
            residual=c_gt['residual'],
            time=c_gt['time']
            )
    print(tgt.__dict__)
    return tgt



# Write cImage data into C structs
def store_image_in_c(img_to, img_from):
    numpy.copyto(img_to.data, img_from.data)
    store_image_pickles(img_to, img_from)

# Phasecentres are too Pythonic to handle right now, so we pickle them
def store_phasecentre(c_phasecentre, phasecentre):
    store_pickle(c_phasecentre, phasecentre, raw_c_ptr=True)

def load_phasecentre(c_phasecentre):
    return load_pickle(c_phasecentre, 4999)

ff.cdef("""
typedef struct {
  char *confname;
  double pc_ra;
  double pc_dec;
  double *times;
  int ntimes;
  double *freqs;
  int nfreqs;
  double *channel_bandwidth;
  int nchanwidth;
  int nbases;
  int nant;
  int npol;
  int nrec;
  double rmax;
  char *polframe;
} ARLConf;
""")

@ff.callback("void (*)(ARLConf *, ARLVis *)")
def arl_create_visibility_ffi(lowconfig, c_res_vis):
    lowcore_name = str(ff.string(lowconfig.confname), 'utf-8')
    lowcore = create_named_configuration(lowcore_name)

    times = numpy.frombuffer(ff.buffer(lowconfig.times, 8*lowconfig.ntimes), dtype='f8', count=lowconfig.ntimes)
    frequency = numpy.frombuffer(ff.buffer(lowconfig.freqs, 8*lowconfig.nfreqs), dtype='f8', count=lowconfig.nfreqs)
    channel_bandwidth = numpy.frombuffer(ff.buffer(lowconfig.channel_bandwidth, 8*lowconfig.nchanwidth), dtype='f8', count=lowconfig.nchanwidth)

    print(lowcore_name)
    print("Times: ", times)
    print("Freqs: ", frequency)
    print("BW : ", channel_bandwidth)
    print("PCentre: ", lowconfig.pc_ra, lowconfig.pc_dec)

    phasecentre = SkyCoord(ra=lowconfig.pc_ra * u.deg, dec=lowconfig.pc_dec*u.deg, frame='icrs', 
            equinox='J2000')
    polframe = str(ff.string(lowconfig.polframe), 'utf-8')

    vt = create_visibility(lowcore, times, frequency,
            channel_bandwidth=channel_bandwidth, weight=1.0,
            phasecentre=phasecentre,
            polarisation_frame=PolarisationFrame(polframe))

    py_res_vis = cARLVis(c_res_vis)

    numpy.copyto(py_res_vis, vt.data)

    store_phasecentre(c_res_vis.phasecentre, phasecentre)

arl_create_visibility=collections.namedtuple("FFIX", "address")
arl_create_visibility.address=int(ff.cast("size_t", arl_create_visibility_ffi))

@ff.callback("void (*)(ARLConf *, ARLVis *)")
def arl_create_blockvisibility_ffi(lowconfig, c_res_vis):
    lowcore_name = str(ff.string(lowconfig.confname), 'utf-8')
    lowcore = create_named_configuration(lowcore_name)

    times = numpy.frombuffer(ff.buffer(lowconfig.times, 8*lowconfig.ntimes), dtype='f8', count=lowconfig.ntimes)
    frequency = numpy.frombuffer(ff.buffer(lowconfig.freqs, 8*lowconfig.nfreqs), dtype='f8', count=lowconfig.nfreqs)
    channel_bandwidth = numpy.frombuffer(ff.buffer(lowconfig.channel_bandwidth, 8*lowconfig.nchanwidth), dtype='f8', count=lowconfig.nchanwidth)

    print(lowcore_name)
    print("Times: ", times)
    print("Freqs: ", frequency)
    print("BW : ", channel_bandwidth)
    print("PCentre: ", lowconfig.pc_ra, lowconfig.pc_dec)

    phasecentre = SkyCoord(ra=lowconfig.pc_ra * u.deg, dec=lowconfig.pc_dec*u.deg, frame='icrs',
            equinox='J2000')

    polframe = str(ff.string(lowconfig.polframe), 'utf-8')
    print("Polarisation frame: ", polframe)
    vt = create_blockvisibility(lowcore, times, frequency=frequency,
            channel_bandwidth=channel_bandwidth, weight=1.0,
            phasecentre=phasecentre,
            polarisation_frame=PolarisationFrame(polframe))

    py_res_vis = cARLBlockVis(c_res_vis, lowconfig.nant, lowconfig.nfreqs)

    numpy.copyto(py_res_vis, vt.data)

    store_phasecentre(c_res_vis.phasecentre, phasecentre)

    receptor_frame = ReceptorFrame(vt.polarisation_frame.type)
    lowconfig.nrec = receptor_frame.nrec

arl_create_blockvisibility=collections.namedtuple("FFIX", "address")
arl_create_blockvisibility.address=int(ff.cast("size_t", arl_create_blockvisibility_ffi))

@ff.callback("void (*)(ARLConf *, const ARLVis *, const ARLVis *, long long int *, ARLVis *)")
def arl_convert_visibility_to_blockvisibility_ffi(lowconfig, vis_in, blockvis_in, cindex_in, blockvis_out):
    lowcore_name = str(ff.string(lowconfig.confname), 'utf-8')
    lowcore = create_named_configuration(lowcore_name)

    times = numpy.frombuffer(ff.buffer(lowconfig.times, 8*lowconfig.ntimes), dtype='f8', count=lowconfig.ntimes)
    frequency = numpy.frombuffer(ff.buffer(lowconfig.freqs, 8*lowconfig.nfreqs), dtype='f8', count=lowconfig.nfreqs)
    channel_bandwidth = numpy.frombuffer(ff.buffer(lowconfig.channel_bandwidth, 8*lowconfig.nchanwidth), dtype='f8', count=lowconfig.nchanwidth)

    cindex_size = lowconfig.nant*lowconfig.nant*lowconfig.nfreqs*lowconfig.ntimes
    py_cindex = numpy.frombuffer(ff.buffer(cindex_in, 8*cindex_size), dtype='int', count=cindex_size)

    c_visin = cARLVis(vis_in)
    py_visin = helper_create_visibility_object(c_visin)
    py_visin.phasecentre = load_phasecentre(vis_in.phasecentre)
    py_visin.configuration = lowcore
    py_visin.cindex = py_cindex
    polframe = str(ff.string(lowconfig.polframe), 'utf-8')
    py_visin.polarisation_frame = PolarisationFrame(polframe)


    c_blockvisin = cARLBlockVis(blockvis_in, lowconfig.nant, lowconfig.nfreqs)
    py_blockvisin = helper_create_blockvisibility_object(c_blockvisin, frequency, channel_bandwidth, lowcore)
    py_blockvisin.phasecentre = load_phasecentre(blockvis_in.phasecentre)
    py_blockvisin.configuration = lowcore
    py_blockvisin.polarisation_frame = PolarisationFrame(polframe)

    py_visin.blockvis = py_blockvisin

    py_blockvisout = convert_visibility_to_blockvisibility(py_visin)

    py_blockvis_out = cARLBlockVis(blockvis_out, lowconfig.nant, lowconfig.nfreqs)
    numpy.copyto(py_blockvis_out, py_blockvisout.data)

arl_convert_visibility_to_blockvisibility=collections.namedtuple("FFIX", "address")
arl_convert_visibility_to_blockvisibility.address=int(ff.cast("size_t", arl_convert_visibility_to_blockvisibility_ffi))


@ff.callback("void (*)(ARLConf *, const ARLVis *, ARLGt *)")
def arl_create_gaintable_from_blockvisibility_ffi(lowconfig, blockvis_in, gt_out):
    lowcore_name = str(ff.string(lowconfig.confname), 'utf-8')
    lowcore = create_named_configuration(lowcore_name)

    times = numpy.frombuffer(ff.buffer(lowconfig.times, 8*lowconfig.ntimes), dtype='f8', count=lowconfig.ntimes)
    frequency = numpy.frombuffer(ff.buffer(lowconfig.freqs, 8*lowconfig.nfreqs), dtype='f8', count=lowconfig.nfreqs)
    channel_bandwidth = numpy.frombuffer(ff.buffer(lowconfig.channel_bandwidth, 8*lowconfig.nchanwidth), dtype='f8', count=lowconfig.nchanwidth)

    c_blockvisin = cARLBlockVis(blockvis_in, lowconfig.nant, lowconfig.nfreqs)
    py_blockvisin = helper_create_blockvisibility_object(c_blockvisin, frequency, channel_bandwidth, lowcore)
    py_blockvisin.phasecentre = load_phasecentre(blockvis_in.phasecentre)
    py_blockvisin.configuration = lowcore
    polframe = str(ff.string(lowconfig.polframe), 'utf-8')
    py_blockvisin.polarisation_frame = PolarisationFrame(polframe)

    py_gt = create_gaintable_from_blockvisibility(py_blockvisin)

#    print(py_gt.data.size, py_gt.data.itemsize)
#    print(py_gt.frequency.size)
#    print(py_gt.receptor_frame)

#    receptor_frame = ReceptorFrame(py_blockvisin.polarisation_frame.type)
#    pframe1 = PolarisationFrame(polframe)
#    recframe1 = ReceptorFrame(pframe1.type) 
#    print(receptor_frame.nrec, recframe1.nrec, lowcore.receptor_frame.nrec)
    c_gt_out = cARLGt(gt_out, lowconfig.nant, lowconfig.nfreqs, lowconfig.nrec)
    numpy.copyto(c_gt_out, py_gt.data)


arl_create_gaintable_from_blockvisibility=collections.namedtuple("FFIX", "address")
arl_create_gaintable_from_blockvisibility.address=int(ff.cast("size_t", arl_create_gaintable_from_blockvisibility_ffi))


@ff.callback("void (*)(ARLConf *, ARLGt *)")
def arl_simulate_gaintable_ffi(lowconfig, gt):
    lowcore_name = str(ff.string(lowconfig.confname), 'utf-8')
    lowcore = create_named_configuration(lowcore_name)

    times = numpy.frombuffer(ff.buffer(lowconfig.times, 8*lowconfig.ntimes), dtype='f8', count=lowconfig.ntimes)
    frequency = numpy.frombuffer(ff.buffer(lowconfig.freqs, 8*lowconfig.nfreqs), dtype='f8', count=lowconfig.nfreqs)

    polframe = str(ff.string(lowconfig.polframe), 'utf-8')
    polarisation_frame = PolarisationFrame(polframe)
    receptor_frame = PolarisationFrame(polframe)

    c_gt = cARLGt(gt, lowconfig.nant, lowconfig.nfreqs, lowconfig.nrec)
    py_gt = helper_create_gaintable_object(c_gt, frequency)
    py_gt.receptor_frame = receptor_frame
    print(py_gt.__dict__)
#    py_gt = simulate_gaintable(py_gt, phase_error = 1.0)

#    numpy.copyto(c_gt, py_gt.data)
    

arl_simulate_gaintable=collections.namedtuple("FFIX", "address")
arl_simulate_gaintable.address=int(ff.cast("size_t", arl_simulate_gaintable_ffi))


ff.cdef("""
typedef struct {
	int vis_slices;
	int npixel;
	double cellsize;
	double guard_band_image;
	double delA;
	int wprojection_planes;
} ARLadvice ;
""")

@ff.callback("void (*)(ARLConf *, ARLVis *, ARLadvice *)")
def arl_advise_wide_field_ffi(lowconfig, vis_in, adv):
    lowcore_name = str(ff.string(lowconfig.confname), 'utf-8')
    lowcore = create_named_configuration(lowcore_name, rmax=lowconfig.rmax)
    c_visin = cARLBlockVis(vis_in, lowconfig.nant, lowconfig.nfreqs)

    frequency = numpy.frombuffer(ff.buffer(lowconfig.freqs, 8*lowconfig.nfreqs), dtype='f8', count=lowconfig.nfreqs)
    channel_bandwidth = numpy.frombuffer(ff.buffer(lowconfig.channel_bandwidth, 8*lowconfig.nchanwidth), dtype='f8', count=lowconfig.nchanwidth)

    py_visin = helper_create_blockvisibility_object(c_visin, frequency, channel_bandwidth, lowcore)
    py_visin.phasecentre = load_phasecentre(vis_in.phasecentre)
    
    polframe = str(ff.string(lowconfig.polframe), 'utf-8')
    py_visin.polarisation_frame = PolarisationFrame(polframe)
    advice=advise_wide_field(py_visin, guard_band_image=adv.guard_band_image, delA=adv.delA,
                             wprojection_planes=adv.wprojection_planes)
    print(advice['vis_slices'], advice['npixels2'], advice['cellsize'])
    
    adv.cellsize = advice['cellsize']
    adv.vis_slices = advice['vis_slices']
    adv.npixel = advice['npixels2']
	
arl_advise_wide_field=collections.namedtuple("FFIX", "address")
arl_advise_wide_field.address=int(ff.cast("size_t", arl_advise_wide_field_ffi))


ff.cdef("""
typedef struct {int nant, nbases;} ant_t;
""")

# Get the number of baselines for the given configuration
@ff.callback("void (*) (char*, ant_t *)")
def helper_get_nbases_ffi(config_name, nbases_in):
    tconfig_name = str(ff.string(config_name), 'utf-8')
    lowcore = create_named_configuration(tconfig_name)
    nbases_in.nant = len(lowcore.xyz)
    nbases_in.nbases = int(len(lowcore.xyz)*(len(lowcore.xyz)-1)/2)
    print(tconfig_name,nbases_in.nant, nbases_in.nbases )

helper_get_nbases=collections.namedtuple("FFIX", "address")    
helper_get_nbases.address=int(ff.cast("size_t", helper_get_nbases_ffi))  

@ff.callback("void (*)(ARLConf *, double, int, int *)")
def helper_get_image_shape_multifreq_ffi(lowconfig, cellsize, npixel, c_shape):
    frequency = numpy.frombuffer(ff.buffer(lowconfig.freqs, 8*lowconfig.nfreqs), dtype='f8', count=lowconfig.nfreqs)
    channel_bandwidth = numpy.frombuffer(ff.buffer(lowconfig.channel_bandwidth, 8*lowconfig.nchanwidth), dtype='f8', count=lowconfig.nchanwidth)
    res = create_low_test_image_from_gleam(npixel=npixel, frequency=frequency,
    channel_bandwidth=channel_bandwidth, cellsize=cellsize, flux_limit = 10.)
#	phasecentre=phasecentre, applybeam=True)
#    res = create_test_image(frequency=frequency, cellsize=cellsize, npixel = npixel)

    shape = list(res.data.shape)
    # TODO fix ugly
    numpy.copyto(numpy.frombuffer(ff.buffer(c_shape,4*4),dtype='i4',count=4), shape)

helper_get_image_shape_multifreq=collections.namedtuple("FFIX", "address")
helper_get_image_shape_multifreq.address=int(ff.cast("size_t", helper_get_image_shape_multifreq_ffi))

# TODO temporary until better solution found
@ff.callback("void (*)(const double *, double, int *)")
def helper_get_image_shape_ffi(freq, cellsize, c_shape):
    res = create_test_image(freq, cellsize)

    shape = list(res.data.shape)
    # TODO fix ugly
    numpy.copyto(numpy.frombuffer(ff.buffer(c_shape,4*4),dtype='i4',count=4), shape)

helper_get_image_shape=collections.namedtuple("FFIX", "address")
helper_get_image_shape.address=int(ff.cast("size_t", helper_get_image_shape_ffi))

# TODO properly implement this routine - shouldn't be within create_test_image
#@ff.callback("void (*)(const ARLVis *, Image *)")
#def helper_set_image_params_ffi(vis, image):
#    phasecentre = load_phasecentre(vis.phasecentre)
#
#    py_image = cImage(image)
#
#    py_image.wcs.wcs.crval[0] = phasecentre.ra.deg
#    py_image.wcs.wcs.crval[1] = phasecentre.dec.deg
#    py_image.wcs.wcs.crpix[0] = float(nx // 2)
#    py_image.wcs.wcs.crpix[1] = float(ny // 2)
#
#helper_set_image_params=collections.namedtuple("FFIX", "address")
#helper_set_image_params.address=int(ff.cast("size_t", helper_set_image_params_ffi))

@ff.callback("void (*)(const double *, double, char*, Image *)")
def arl_create_test_image_ffi(frequency, cellsize, c_phasecentre, out_img):
    py_outimg = cImage(out_img, new=True)

    res = create_test_image(frequency, cellsize)

    phasecentre = load_phasecentre(c_phasecentre)

    nchan, npol, ny, nx = res.data.shape

    res.wcs.wcs.crval[0] = phasecentre.ra.deg
    res.wcs.wcs.crval[1] = phasecentre.dec.deg
    res.wcs.wcs.crpix[0] = float(nx // 2)
    res.wcs.wcs.crpix[1] = float(ny // 2)

    store_image_in_c(py_outimg, res)

arl_create_test_image=collections.namedtuple("FFIX", "address")
arl_create_test_image.address=int(ff.cast("size_t", arl_create_test_image_ffi))

@ff.callback("void (*)(ARLConf *, double, int, char*, Image *)")
def arl_create_low_test_image_from_gleam_ffi(lowconfig, cellsize, npixel, c_phasecentre, out_img):
    py_outimg = cImage(out_img, new=True)
    frequency = numpy.frombuffer(ff.buffer(lowconfig.freqs, 8*lowconfig.nfreqs), dtype='f8', count=lowconfig.nfreqs)
    channel_bandwidth = numpy.frombuffer(ff.buffer(lowconfig.channel_bandwidth, 8*lowconfig.nchanwidth), dtype='f8', count=lowconfig.nchanwidth)

    phasecentre = load_phasecentre(c_phasecentre)
    res = create_low_test_image_from_gleam(npixel=npixel, frequency=frequency,
    channel_bandwidth=channel_bandwidth, cellsize=cellsize, flux_limit = 1.0, phasecentre=phasecentre, applybeam=True)

    nchan, npol, ny, nx = res.data.shape

    res.wcs.wcs.crval[0] = phasecentre.ra.deg
    res.wcs.wcs.crval[1] = phasecentre.dec.deg
    res.wcs.wcs.crpix[0] = float(nx // 2)
    res.wcs.wcs.crpix[1] = float(ny // 2)

    store_image_in_c(py_outimg, res)

arl_create_low_test_image_from_gleam=collections.namedtuple("FFIX", "address")
arl_create_low_test_image_from_gleam.address=int(ff.cast("size_t", arl_create_low_test_image_from_gleam_ffi))


@ff.callback("void (*)(const ARLVis *, const Image *, ARLVis *)")
def arl_predict_2d_ffi(vis_in, img, vis_out):
    c_visin = cARLVis(vis_in)
    py_visin = helper_create_visibility_object(c_visin)
    c_img = cImage(img)

    
    py_visin.phasecentre = load_phasecentre(vis_in.phasecentre)

    res = predict_2d(py_visin, c_img)

    vis_out.nvis = vis_in.nvis
    vis_out.npol = vis_in.npol
    c_visout = cARLVis(vis_out)

    numpy.copyto(c_visout, res.data)
    store_phasecentre(vis_out.phasecentre, res.phasecentre)

    #arl_copy_visibility(py_visin, c_visout, False)

arl_predict_2d=collections.namedtuple("FFIX", "address")
arl_predict_2d.address=int(ff.cast("size_t", arl_predict_2d_ffi))


@ff.callback("void (*)(ARLConf *, const ARLVis *, const Image *, ARLVis *, ARLVis *, long long int *)")
def arl_predict_function_ffi(lowconfig, vis_in, img, vis_out, blockvis_out, cindex_out):

    lowcore_name = str(ff.string(lowconfig.confname), 'utf-8')
    lowcore = create_named_configuration(lowcore_name, rmax=lowconfig.rmax)

    cindex_size = lowconfig.nant*lowconfig.nant*lowconfig.nfreqs*lowconfig.ntimes
    py_cindex = numpy.frombuffer(ff.buffer(cindex_out, 8*cindex_size), dtype='int', count=cindex_size)

    c_visin = cARLBlockVis(vis_in, lowconfig.nant, lowconfig.nfreqs)
    frequency = numpy.frombuffer(ff.buffer(lowconfig.freqs, 8*lowconfig.nfreqs), dtype='f8', count=lowconfig.nfreqs)
    channel_bandwidth = numpy.frombuffer(ff.buffer(lowconfig.channel_bandwidth, 8*lowconfig.nchanwidth), dtype='f8', count=lowconfig.nchanwidth)
    py_visin = helper_create_blockvisibility_object(c_visin, frequency, channel_bandwidth, lowcore)
    c_img = cImage(img)
    py_visin.phasecentre = load_phasecentre(vis_in.phasecentre)
    polframe = str(ff.string(lowconfig.polframe), 'utf-8')
    py_visin.polarisation_frame = PolarisationFrame(polframe)

    res = predict_function(py_visin, c_img, vis_slices=51, context='wstack')
#    print(sys.getsizeof(py_visin.data), sys.getsizeof(res.data), sys.getsizeof(res.blockvis.data))
#    print(type(res.cindex), type(res.cindex[0]), len(res.cindex))
#    print(sys.getsizeof(res.cindex))

    vis_out.npol = vis_in.npol
    c_visout = cARLVis(vis_out)
    numpy.copyto(c_visout, res.data)
    store_phasecentre(vis_out.phasecentre, res.phasecentre)
    numpy.copyto(py_cindex, res.cindex)

    py_blockvis_out = cARLBlockVis(blockvis_out, lowconfig.nant, lowconfig.nfreqs)
    numpy.copyto(py_blockvis_out, res.blockvis.data)
    store_phasecentre(blockvis_out.phasecentre, res.phasecentre)

arl_predict_function=collections.namedtuple("FFIX", "address")
arl_predict_function.address=int(ff.cast("size_t", arl_predict_function_ffi))

@ff.callback("void (*)(ARLConf *, ARLVis *, const Image *)")
def arl_predict_function_blockvis_ffi(lowconfig, vis_in, img):

    lowcore_name = str(ff.string(lowconfig.confname), 'utf-8')
    lowcore = create_named_configuration(lowcore_name, rmax=lowconfig.rmax)

    c_visin = cARLBlockVis(vis_in, lowconfig.nant, lowconfig.nfreqs)
    frequency = numpy.frombuffer(ff.buffer(lowconfig.freqs, 8*lowconfig.nfreqs), dtype='f8', count=lowconfig.nfreqs)
    channel_bandwidth = numpy.frombuffer(ff.buffer(lowconfig.channel_bandwidth, 8*lowconfig.nchanwidth), dtype='f8', count=lowconfig.nchanwidth)
    py_visin = helper_create_blockvisibility_object(c_visin, frequency, channel_bandwidth, lowcore)
    c_img = cImage(img)
    py_visin.phasecentre = load_phasecentre(vis_in.phasecentre)

    res = predict_function(py_visin, c_img, vis_slices=51, context='wstack')
    py_blockvis = convert_visibility_to_blockvisibility(res)

    c_visin = cARLBlockVis(vis_in, lowconfig.nant, lowconfig.nfreqs)

    numpy.copyto(c_visin, py_blockvis.data)
#    store_phasecentre(vis_out.phasecentre, res.phasecentre)

arl_predict_function_blockvis=collections.namedtuple("FFIX", "address")
arl_predict_function_blockvis.address=int(ff.cast("size_t", arl_predict_function_blockvis_ffi))


def store_image_in_c_2(img_to, img_from):
    numpy.copyto(img_to.data, img_from.data)
    store_image_pickles(img_to, img_from)

@ff.callback("void (*)(const ARLVis *, const Image *, bool dopsf, Image *, double *)")
def arl_invert_2d_ffi(invis, in_image, dopsf, out_image, sumwt):
    py_visin = helper_create_visibility_object(cARLVis(invis))
    c_in_img = cImage(in_image)
    c_out_img = cImage(out_image, new=True)
    py_visin.phasecentre = load_phasecentre(invis.phasecentre)

    if dopsf:
        out, sumwt = invert_2d(py_visin, c_in_img, dopsf=True)
    else:
        out, sumwt = invert_2d(py_visin, c_in_img)


    store_image_in_c_2(c_out_img, out)

arl_invert_2d=collections.namedtuple("FFIX", "address")
arl_invert_2d.address=int(ff.cast("size_t", arl_invert_2d_ffi))

@ff.callback("void (*)(const ARLVis *, Image *)")
def arl_create_image_from_visibility_ffi(vis_in, img_in):
    c_vis = cARLVis(vis_in)
    c_img = cImage(img_in, new=True);

    # We need a proper Visibility object - not this, and not a cARLVis
    # This is temporary - just so we have some data to pass to
    # the create_... routine
    tvis = helper_create_visibility_object(c_vis)
    tvis.phasecentre = load_phasecentre(vis_in.phasecentre)

    # Default args for now
    image = create_image_from_visibility(tvis, cellsize=0.001, npixel=256)

    #numpy.copyto(c_img.data, image.data)

    # Pickle WCS and polframe, until better way is found to handle these data
    # structures
    #store_image_pickles(c_img, image)
    store_image_in_c(c_img, image)


arl_create_image_from_visibility=collections.namedtuple("FFIX", "address")    
arl_create_image_from_visibility.address=int(ff.cast("size_t",
    arl_create_image_from_visibility_ffi))    

@ff.callback("void (*)(Image *, Image *, Image *, Image *)")
def arl_deconvolve_cube_ffi(dirty, psf, restored, residual):
    c_dirty = cImage(dirty)
    c_psf = cImage(psf)
    c_residual = cImage(residual, new=True)
    c_restored = cImage(restored, new=True)

    py_restored, py_residual = deconvolve_cube(c_dirty, c_psf,
            niter=1000,threshold=0.001, fracthresh=0.01, window_shape='quarter',
            gain=0.7, scales=[0,3,10,30])

    store_image_in_c(c_restored,py_restored)
    store_image_in_c(c_residual,py_residual)


arl_deconvolve_cube=collections.namedtuple("FFIX", "address")    
arl_deconvolve_cube.address=int(ff.cast("size_t", arl_deconvolve_cube_ffi))    

@ff.callback("void (*)(Image *, Image *, Image*, Image*)")
def arl_restore_cube_ffi(model, psf, residual, restored):
    # Cast C Image structs to Python objects
    c_model = cImage(model)
    c_psf = cImage(psf)
    if residual:
        c_residual = cImage(residual)
    else:
        c_residual = None
    c_restored = cImage(restored, new=True)

    # Calculate
    py_restored = restore_cube(c_model, c_psf, c_residual)

    # Copy Python result to C result struct
    store_image_in_c(c_restored,py_restored)

arl_restore_cube=collections.namedtuple("FFIX", "address")    
arl_restore_cube.address=int(ff.cast("size_t", arl_restore_cube_ffi))    



