# Author: Bojan Nikolic <b.nikolic@mrao.cam.ac.uk>
# ARL Wrapping

import cffi
import numpy
import collections

import arl
from arl.visibility.base import copy_visibility

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
    print (ARLDataVisSize(nvis, npol))
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
    tvis=arl.visibility.base.copy_visibility(nvisin, zero=zero)

    # Copy the result into the output buffer
    visout.npol=visin.npol
    visout.nvis=visin.nvis
    nvisout=cARLVis(visout)
    numpy.copyto(nvisout, tvis)


arl_copy_visibility=collections.namedtuple("FFIX", "address")    
arl_copy_visibility.address=int(ff.cast("size_t", arl_copy_visibility_ffi))    

    
