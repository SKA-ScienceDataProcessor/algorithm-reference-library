# Author: Bojan Nikolic <b.nikolic@mrao.cam.ac.uk>
# ARL Wrapping

import cffi
import numpy
import collections

from astropy.coordinates import SkyCoord
from astropy import units as u

#import arl
from arl.visibility.base import create_visibility, copy_visibility
from arl.data.data_models import Image, Visibility, BlockVisibility
from arl.image.deconvolution import deconvolve_cube, restore_cube
from arl.imaging.base import create_image_from_visibility, predict_2d
from arl.util.testing_support import create_named_configuration, create_test_image
from arl.data.polarisation import PolarisationFrame

from arl.imaging import invert_2d

import pickle

ff = cffi.FFI()

ff.cdef("""
typedef struct {
  size_t nvis;
  int npol;
  void *data;
} ARLVis;
""")

def ARLDataVisSize(nvis, npol):
    return (72+32*int(npol))*int(nvis)

def cARLVis(visin):
    """
    Convert a const ARLVis * into the ARL Visiblity structure
    """
    npol=visin.npol
    nvis=visin.nvis
    #print (ARLDataVisSize(nvis, npol))
    desc = [('uvw', '>f8', (3,)),
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
    new_image.data = new_image.data.reshape(data_shape)
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

def store_image_pickles(c_img, py_img):
    wcs_pickle = pickle.dumps(py_img.wcs)
    wcs_buf = numpy.frombuffer(wcs_pickle, dtype='b', count=len(wcs_pickle))
    polframe_pickle = pickle.dumps(py_img.polarisation_frame)
    polframe_buf = numpy.frombuffer(polframe_pickle, dtype='b',
        count=len(polframe_pickle))

    numpy.copyto(c_img.wcs, wcs_buf)
    numpy.copyto(c_img.polarisation_frame, polframe_buf)

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
def helper_create_blockvisibility_object(c_vis):
    # This may be incorrect
    # especially the data field...
    tvis= BlockVisibility(
            data=c_vis,
            frequency=c_vis['frequency'],
            channel_bandwidth=c_vis['channel_bandwidth'],
            integration_time=c_vis['integration_time'],
            weight=c_vis['weight'],
            uvw=c_vis['uvw'],
            time=c_vis['time']
            )
    return tvis

def store_image_in_c(img_to, img_from):
    numpy.copyto(img_to.data, img_from.data)
    store_image_pickles(img_to, img_from)

@ff.callback("void (*)(const char *, double*, double*, double*, ARLVis *)")
def arl_create_visibility_ffi(lowcore_name, c_times, c_frequency,
        channel_bandwidth, c_res_vis):
    lowcore_name = str(ff.string(lowcore_name), 'utf-8')
    lowcore = create_named_configuration(lowcore_name)

    times = numpy.frombuffer(ff.buffer(c_times, 8), dtype='f8', count=1)
    frequency = numpy.frombuffer(ff.buffer(c_frequency, 8), dtype='f8', count=1)

    phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-45.0*u.deg, frame='icrs',
            equinox='J200')

    vt =create_visibility(lowcore, times, frequency,
            channel_bandwidth=channel_bandwidth, weight=1.0,
            phasecentre=phasecentre,
            polarisation_frame=PolarisationFrame('stokesI'))

    py_res_vis = cARLVis(c_res_vis)

    #print(py_res_vis.dtype)
    #print(vt.data.dtype)
    numpy.copyto(py_res_vis, vt.data)

arl_create_visibility=collections.namedtuple("FFIX", "address")
arl_create_visibility.address=int(ff.cast("size_t", arl_create_visibility_ffi))

# TODO temporary until better solution found
@ff.callback("void (*)(const double *, double, int *)")
def helper_get_image_shape_ffi(freq, cellsize, c_shape):
    res = create_test_image(freq, cellsize)

    shape = list(res.data.shape)
    # TODO fix ugly
    numpy.copyto(numpy.frombuffer(ff.buffer(c_shape,4*4),dtype='i4',count=4), shape)

helper_get_image_shape=collections.namedtuple("FFIX", "address")
helper_get_image_shape.address=int(ff.cast("size_t", helper_get_image_shape_ffi))

@ff.callback("void (*)(const double *, double, Image *)")
def arl_create_test_image_ffi(frequency, cellsize, out_img):
    py_outimg = cImage(out_img, new=True)

    res = create_test_image(frequency, cellsize)

    nchan, npol, ny, nx = res.data.shape
    phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-45.0*u.deg, frame='icrs',
            equinox='J200')

    res.wcs.wcs.crval[0] = phasecentre.ra.deg
    res.wcs.wcs.crval[1] = phasecentre.dec.deg
    res.wcs.wcs.crpix[0] = float(nx // 2)
    res.wcs.wcs.crpix[1] = float(ny // 2)

    store_image_in_c(py_outimg, res)

arl_create_test_image=collections.namedtuple("FFIX", "address")
arl_create_test_image.address=int(ff.cast("size_t", arl_create_test_image_ffi))

@ff.callback("void (*)(const ARLVis *, const Image *, ARLVis *)")
def arl_predict_2d_ffi(vis_in, img, vis_out):
    c_visin = cARLVis(vis_in)
    py_visin = helper_create_visibility_object(c_visin)
    c_img = cImage(img)

    phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-45.0*u.deg, frame='icrs',
            equinox='J200')
    py_visin.phasecentre = phasecentre

    res = predict_2d(py_visin, c_img)

    vis_out.nvis = vis_in.nvis
    vis_out.npol = vis_in.npol
    c_visout = cARLVis(vis_out)

    numpy.copyto(c_visout, res.data)

    #arl_copy_visibility(py_visin, c_visout, False)

arl_predict_2d=collections.namedtuple("FFIX", "address")
arl_predict_2d.address=int(ff.cast("size_t", arl_predict_2d_ffi))

def store_image_in_c_2(img_to, img_from):
    numpy.copyto(img_to.data, img_from.data)
    store_image_pickles(img_to, img_from)

@ff.callback("void (*)(const ARLVis *, const Image *, bool dopsf, Image *, double *)")
def arl_invert_2d_ffi(invis, in_image, dopsf, out_image, sumwt):
    py_visin = helper_create_visibility_object(cARLVis(invis))
    c_in_img = cImage(in_image)
    c_out_img = cImage(out_image, new=True)
    phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-45.0*u.deg, frame='icrs',
            equinox='J200')
    py_visin.phasecentre = phasecentre

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
    #print(type(tvis))
    phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-45.0*u.deg, frame='icrs',
            equinox='J200')
    tvis.phasecentre = phasecentre

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
    #if residual:
    #    c_residual = cImage(residual, new=True)
    #else:
    #    c_residual = None
    c_residual = cImage(residual, new=True)
    c_restored = cImage(restored, new=True)

    py_restored, py_residual = deconvolve_cube(c_dirty, c_psf,
            niter=1000,threshold=0.001, fracthresh=0.01, window_shape='quarter',
            gain=0.7, scales=[0,3,10,30])

    #restored.data[0][0][0][0] = 1111.2222

    #numpy.copyto(c_restored.data,pyrestored.data)
    #numpy.copyto(c_residual.data,residual.data)

    store_image_in_c(c_restored,py_restored)
    store_image_in_c(c_residual,py_residual)

    # This was for testing only
    #store_image_pickles(c_restored, restored)

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
    #restored.data[0][0][0][0] = 1337.7331

    # Copy Python result to C result struct
    store_image_in_c(c_restored,py_restored)

arl_restore_cube=collections.namedtuple("FFIX", "address")    
arl_restore_cube.address=int(ff.cast("size_t", arl_restore_cube_ffi))    

