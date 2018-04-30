
Effective April 30 2018, the ARL has been updated to be consistent with the SDP Processing Architecture. This required 
very substantial changes throughout. The code is consistent internally but ARL code kept outside the code tree will 
need to be updated manually.

* The top level directory arl has been split into three: libs, processing_components, and workflows
    - libs contains functions that are not accessed directly by the Execution Framework
    - processing_components contains functions that may be accessed by the EF. 
    - workflows contains high level workflows using the processing_components. This eventually will migrate to the EF
     but some are kept here as scripts or notebooks.
* The tests and notebooks have been moved to be inside the appropriate directory.
* The data definitions formerly in arl/data have been moved to a top level directory data_models. 
* The top level Makefile has been updated
* The docs have been updated
* The use of the term 'graph' has been replaced in many places by 'list' to reflect the wrapping of dask in 
arlexecute.

 