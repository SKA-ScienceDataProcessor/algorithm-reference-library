# ARL C API (work in progress)
In this folder contains a proof-of-concept C interface wrapping Python ARL
routines, and a simple pipeline implementation in C.

# Rationale
We want to be able to perform testing of different Execution Frameworks, without
having to implement computational routines for every language. By implementing
an interface in C, we can call the ARL routines from basically any language.
This allows us to focus our time on testing the actual EFs.

# starpu\_timg/timg\_starpu.c

Contains a (partial) implementation of the TIMG pipeline from the ffi\_demo using
STARPU. This is intended to demo a more 'real-world' framework with the CFFI
interface to the ARL routines.

This adds a requirement for StarPU to the build process. To disable, simply
comment out the timg\_starpu related lines from `setup.py` and `Makefile`.

# ICAL ffi demo
This example will repeat a functionality of the python script `src/imaging-pipelines-sip.py`
which calls a non-Dask version of ARL ICAL pipeline.

# Building
Modify `setup.py` and `Makefile` to point to the correct installation directory
for StarPU if necessary.

Then, simply run `python3 setup.py build`
Alternatively, use `CFLAGS` environment variable, e.g.
`CFLAGS="-I/usr/local/cuda/include/ -I/usr/include/cfitsio/" python3 setup.py build`
In this example CUDA is required by StarPU.


# How does it work
CFFI provides interface to call C from python.

We import arlwrap.py module using PyObjects, then we use ffi callback mechanism (ABI mode)
to efectively call our arl python code from C. 

The core mechanism is bk_getfn() function which using PyObjects imports arlwrap.py and 
returns the pointer to the python function (in arlwrap.py) as a callable C function, plus 
calls the function --only this function is not in arlwrap.py as such, it is a tuple.

arlwrap.c 
========
For every function it calls bk_getfn() 

arlwrap.py
=========
Sets up the FFI callback to call the python code in ARL.



# TODOs
 - Currently, we are pickling some metadata and storing it in a character
	 array. However, we don't know the size of the pickled objects in advance, so
	 we are using hard-coded sizes determined by trial and error. This needs to be
	 resolved.
 - Implement interfaces for additional routines
 - Remove some duplication
 - ...

