# Author: Bojan Nikolic <b.nikolic@mrao.cam.ac.uk>
# ARL Wrapping

import cffi

ff = cffi.FFI()

# NB CFFI does not understand float complex *
ff.cdef("""
typedef struct {
  size_t nvis;
  // Shape: (3, nvis)
  double *uvw;
  double *time;
  double *freq;
  double *bw;
  double *intgt;
  int *a1;
  int *a2;
  float *cwis;
  float *wght;
  float *imgwght;
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
    
