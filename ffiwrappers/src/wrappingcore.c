#include <Python.h> 

#include "../include/wrappingcore.h"

#ifdef __GNUC__
#define ARL_TLS __thread
#else
#define ARL_TLS
#endif

static ARL_TLS int arl_bypass_ = 0;
static ARL_TLS int arl_bypass_check_ = 0;

void void_routine() {
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
		return (size_t)&void_routine;
	}

  PyGILState_STATE gilstate = PyGILState_Ensure();

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
