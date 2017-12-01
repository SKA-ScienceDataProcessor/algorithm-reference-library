// Author: Bojan Nikolic <b.nikolic@mrao.cam.ac.uk>
//
// Define symbols for the declared wrapper functions -- the definition
// simply fetches the pointer from Python and calls that.
//

#include <Python.h>

#include "arlwrap.h"

size_t bk_getfn(const char* fname)
{
  size_t res=0;
  PyGILState_STATE gilstate = PyGILState_Ensure();
  
  PyObject *m, *pyfn, *fnaddress;
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


// Not a prototype, so warning will be generated. TODO: switch off
// warning for this instance only?
#define BKFNPY(F)  (* ( void (*)() )(bk_getfn( #F ))) 



void arl_copy_visibility(const ARLVis *visin,
			 ARLVis *visout,
			 bool zero)
{
  BKFNPY(arl_copy_visibility)(visin, visout, zero);
}
