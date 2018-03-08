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


# TODOs
 - Currently, we are pickling some metadata and storing it in a character
	 array. However, we don't know the size of the pickled objects in advance, so
	 we are using hard-coded sizes determined by trial and error. This needs to be
	 resolved.
 - Implement interfaces for additional routines
 - Remove some duplication
 - ...

