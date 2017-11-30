# Author: Bojan Nikolic <b.nikolic@mrao.cam.ac.uk>
# ARL Wrapping

import cffi

ff = cffi.FFI()

# NB CFFI does not understand float complex *
ff.cdef("""
typedef struct {
  size_t nvis;
  int npol;
  void *data;
} ARLVis;
""")

def cARLVis(visin):
    """
    Convert a const ARLVis * into the ARL Visiblity structure
    """
    to do...
    pass

@ff.callback("void (*)(const ARLVis *, ARLVis *, bool)")
def copy_visibility(visin, visout, zero):
    """
    Wrap of arl.visibility.base.copy_visibility
    """
    # This needs implementation
    visin=cARLVis(visin)
    # Now wrap here!
    
