// Author: Bojan Nikolic <b.nikolic@mrao.cam.ac.uk>
//
// Define symbols for the declared wrapper functions -- the definition
// simply fetches the pointer from Python and calls that.
//

#include <stdlib.h>
#include <string.h>
#include <Python.h>

#include "arlwrap.h"

#ifdef __GNUC__
#define ARL_TLS __thread
#else
#define ARL_TLS
#endif
static ARL_TLS int arl_bypass_ = 0;
static ARL_TLS int arl_bypass_check_ = 0;

void void_routine(__VA_ARGS__) {
}
// in: fname: same as the *NamedTuple* containing the FFI memory address of a
// Python routine.
// E.G.: Python routine 'arl_restore_cube_fii', NamedTuple 'arl_restore_cube' =>
// => fname == "arl_restore_cube".
//
// This routine imports the arlwrap.py module, loads the NamedTuple named
// 'fname', and extracts the function address. The address is then returned as
// a callable C function pointer.
size_t bk_getfn(const char* fname)
{
  size_t res=0;
  PyObject *m, *pyfn, *fnaddress;
  PyGILState_STATE gilstate = PyGILState_Ensure();
  
	
  /* Return immediately if the environment variable
   * has already been checked. */
  if (!arl_bypass_check_) {
      char* flag = getenv("ARL_BYPASS_FFI");
      arl_bypass_check_ = 1;
      if (flag && (
              !strcmp(flag, "TRUE") ||
              !strcmp(flag, "true") ||
              !strcmp(flag, "1") ))
          arl_bypass_ = 1;
  }
	// call 'void' routine if ARL_BYPASS_FFI is set
	if (arl_bypass_) {
		return &void_routine;
	}

  if(!(  m= PyImport_ImportModule("arlwrap") ))
    goto failed;
  if(!( pyfn = PyObject_GetAttrString(m, fname)))
    goto failed;
  if(!( fnaddress = PyObject_GetAttrString(pyfn, "address")))
    goto failed;

  res=PyNumber_AsSsize_t(fnaddress, NULL);

  PyGILState_Release(gilstate);
  return res;

 failed:
  PyErr_Print();
  PyGILState_Release(gilstate);  
  return 0;

  // Not decrementing references here since unknown when the objects
  // will be used for C layer. Therefore breakin python functions will
  // never get garbage collected. 
}

void arl_initialize(void)
{
  Py_Initialize();
  PyEval_InitThreads();
}


// Not a prototype, so warning will be generated. TODO: switch off
// warning for this instance only?
#define BKFNPY(F)  (* ( void (*)() )(bk_getfn( #F ))) 


// NB: The CFFI acquires GIL
void helper_get_image_shape(const double *frequency, double cellsize,
		int *shape)
{
        BKFNPY(helper_get_image_shape)(frequency, cellsize, shape);
}

void helper_set_image_params(const ARLVis *vis, Image *image) {
	BKFNPY(helper_set_image_params)(vis, image);
}

void arl_invert_2d(const ARLVis *visin, const Image *img_in, bool dopsf, Image *out, double *sumwt)
{
	BKFNPY(arl_invert_2d)(visin, img_in, dopsf, out, sumwt);
}

void arl_create_visibility(ARLConf *lowconf, ARLVis *res_vis)
{
	BKFNPY(arl_create_visibility)(lowconf, res_vis);
}


void arl_create_test_image(const double *frequency, double cellsize, char *phasecentre,
		Image *res_img)
{
	BKFNPY(arl_create_test_image)(frequency, cellsize, phasecentre, res_img);
}

void arl_copy_visibility(const ARLVis *visin,
			 ARLVis *visout,
			 bool zero)
{
  BKFNPY(arl_copy_visibility)(visin, visout, zero);
}

void arl_predict_2d(const ARLVis *visin, const Image *img, ARLVis *visout) {
	BKFNPY(arl_predict_2d)(visin, img, visout);
}

void arl_create_image_from_visibility(const ARLVis *vis, Image *model) {
	BKFNPY(arl_create_image_from_visibility)(vis, model);
}

void arl_deconvolve_cube(Image *dirty, Image *psf, Image *restored, Image *residual)
{
  BKFNPY(arl_deconvolve_cube)(dirty, psf, restored, residual);
}

void arl_restore_cube(Image *model, Image *psf, Image *residual, Image *restored)
{
  BKFNPY(arl_restore_cube)(model, psf, residual, restored);
}

void helper_get_nbases(char * config_name, ant_t * nbases)
{
  BKFNPY(helper_get_nbases)(config_name, nbases);
}



