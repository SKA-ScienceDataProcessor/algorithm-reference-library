# Author: Bojan Nikolic <b.nikolic@mrao.cam.ac.uk>
# ARL Wrapping

import cffi
import numpy

from data_models.memory_data_models import Image, Visibility, BlockVisibility, GainTable

import pickle

ff = cffi.FFI()



def ARLDataVisSize(nvis, npol):
    return (80+32*int(npol))*int(nvis)

def cARLVis(visin):
    """
    Convert a const ARLVis * into the ARL Visiblity structure
    """
    npol=visin.npol
    nvis=visin.nvis
    print (ARLDataVisSize(nvis, npol))
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
    return (24+24*int(nants*nants) + 32*int(nants*nants)*int(nchan)*int(npol))*int(ntimes)
#    return (24+24*int(nants*nants) + 32*int(nants*nants)*int(nchan)*int(npol))*int(ntimes)

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
            ('weight', '>f8', (nants, nants, nchan, npol)),
            ('imaging_weight', '>f8', (nants, nants, nchan, npol))]
    r=numpy.frombuffer(ff.buffer(visin.data,
                                 ARLBlockDataVisSize(ntimes, nants, nchan, npol)),
                                 dtype=desc,
                                 count=ntimes)
    return r

def ARLDataGTSize(ntimes, nants, nchan, nrec):
    return (8 + 8*nchan*nrec*nrec + 3*8*nants*nchan*nrec*nrec + 8)*ntimes

def cARLGt(gtin, nants, nchan, nrec):
    """
    Convert a const ARLGt * into the ARL GainTable structure 
    """
    ntimes=gtin.nrows
    desc = [('gain', '>c16', (nants, nchan, nrec, nrec)),
            ('weight', '>f8', (nants, nchan, nrec, nrec)),
            ('residual', '>f8', (nchan, nrec, nrec)),
            ('time', '>f8'),
            ('interval', '>f8')]
    r=numpy.frombuffer(ff.buffer(gtin.data,
                                 ARLDataGTSize(ntimes, nants, nchan, nrec)),
                                 dtype=desc,
                                 count=ntimes)
    return r

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
def helper_create_visibility_object(c_vis, config):
    # This may be incorrect
    # especially the data field...
    tvis= Visibility(
            data=c_vis,
            configuration = config,
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
def helper_create_gaintable_object(c_gt, freqs, recframe):
    tgt= GainTable(
            data=None,
	    frequency = freqs,
            gain=c_gt['gain'],
            weight=c_gt['weight'],
            residual=c_gt['residual'],
            time=c_gt['time'],
            interval=c_gt['interval'],
            receptor_frame=recframe
            )
#    print(tgt.__dict__)
    return tgt


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
            image_in.polarisation_frame, 117),
            dtype='b',
            count=117)
    else:
        new_image.wcs = pickle.loads(ff.buffer(image_in.wcs, 2996))
        new_image.polarisation_frame = pickle.loads(ff.buffer(image_in.polarisation_frame,117))
    
    return new_image




# Write cImage data into C structs
def store_image_in_c(img_to, img_from):
    numpy.copyto(img_to.data, img_from.data)
    store_image_pickles(img_to, img_from)

# Phasecentres are too Pythonic to handle right now, so we pickle them
def store_phasecentre(c_phasecentre, phasecentre):
    store_pickle(c_phasecentre, phasecentre, raw_c_ptr=True)

def load_phasecentre(c_phasecentre):
    return load_pickle(c_phasecentre, 4999)

def store_image_in_c_2(img_to, img_from):
    numpy.copyto(img_to.data, img_from.data)
    store_image_pickles(img_to, img_from)



