**September 25, 2018** [Tim] Gridding/degridding code has been refactored to a cleaner design. It also allows more 
control of AW projection.

**September 18, 2018** [Rodrigo] Fix for daliuge port execution of list, to match dask.delayed capabilities. Need to
update version of daliuge:

    pip install "daliuge>=0.5.3" pytest


**September 16, 2018** [Tim] Deleted imaging-pipelines-sip.ipynb. Use imaging-pipelines-serial.ipynb instead.

**September 4, 2018** [Rodrigo]  Added add support for daliuge as an (experimental) backend of the arlexecute module. 
Support uses daliuge's delayed function, which accepts the same parameters as dask's; therefore the change is simple, 
and transparent to the rest of the ARL code. All these changes are within the context of [SDP ticket 
TSK-2569](https://jira.ska-sdp.org/browse/TSK-2569). See also the 
[Integration of ARL with the DALiuGE Execution Framework, part 2](https://confluence.ska-sdp.org/display/WBS/Integration+of+ARL+with+the+DALiuGE+Execution+Framework%2C+part+2)

**August 16, 2018** [Tim] More refactoring to being closer alignment with SDP architecture.
1. There are now wrappers for all processing components, both serial and 
arlexecute. At the moment, these are just pass-throughs but the point is that they can
be expanded as appropriate. The non-python wrappers will be more substantial.
2. There are only workflows for calibration, imaging, and pipelines.
3. To distinguish the nature of the workflows, these are now all called something like
predict_list_arlexecute_workflow since they all work on lists of data models rather 
than just data models.
4. The workflows for serial and arlexecute should work alike. For example, all now 
expect lists of Data Models. This is compared to processing_components
where only single Data Models are accepted. A necessary consequence is
that the full range of imaging algorithms are only available via 
workflows, either as serial or arlexecute versions (and soon other
types of wrappers).
5. libs has been renamed to processing_library.

All Dask/arlexecute code now lives in either wrappers or workflows.

![ARL Module View](./docs/ARL_Module_View.png)


**July 26, 2018** [Tim], Extracted pure-serial uses of processing components 
into workflows/serial (in analogy with workflows/arlexecute). This means that
all functions remaining in processing components are suitable for use in
workflows. The split between processing components and workflows is clearer.
As a consequence nearly all notebooks have moved to workflows/notebooks.

**July 24, 2018** [Tim], Renamed calskymodel to modelpartition to 
be in line with the SDP model views. Also documentation cleanup.

**June 15, 2018** [Tim], Some more moves and renaming:
* processing_components/component_support->libs/execution support
* processing_components/util->processing_components/simulation

generic functions moved to image_components and visibility_components

**June 15 2018** [Tim], the capabilities for reading measurement sets have been improved.
* Both BlockVisibility's and Visibility's can be created. The former is preferred.
* A channel range e.g. range(17,32) can be specified.
* See tests/processing_components/test_visibility_ms for various ways to use this capability.

**June 14, 2018 [Tim]**, BufferDataModel has been introduced as the root of e.g. BufferImage, BufferSkyModel. All of 
these, except for BufferImage use ad hoc HDF5 files. Image can use fits.

**June 12, 2018 [Tim]**, To fill out the architecture, there is now a ProcessingComponentInterface function for executing 
some components. Components have to be wrapped by hand, and the interface defined via a JSON file.

**May 25, 2018** [Piers], Kubernetes support added.

**April 30 2018** [Tim], the ARL has been updated to be consistent with the SDP Processing Architecture. This required 
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

**April 18, 2018** [Tim], Deconvolution can now be done using overlapped, tapered sub-images (aka facets).
Look for deconvolve_facets, deconvolve_overlap, and deconvolve_taper arguments.

