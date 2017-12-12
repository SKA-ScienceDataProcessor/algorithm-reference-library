# ARL C API (work in progress)
In this folder contains a proof-of-concept C interface wrapping Python ARL
routines, and a simple pipeline implementation in C.

# Rationale
We want to be able to perform testing of different Execution Frameworks, without
having to implement computational routines for every language. By implementing
an interface in C, we can call the ARL routines from basically any language.
This allows us to focus our time on testing the actual EFs.

# TODOs
 - Currently, we are pickling some metadata and storing it in a character
	 array. However, we don't know the size of the pickled objects in advance, so
	 we are using hard-coded sizes determined by trial and error. This needs to be
	 resolved.
 - Implement interfaces for additional routines
 - Remove some duplication
 - ...
