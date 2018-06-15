
.. toctree::
   :maxdepth: 2

ARL structure
=============

The ARL structure maps that used in the SDP processing architecture:

.. image:: ./SDP_processing_architecture.png
   :scale: 100 %

Not all connections are shown.

The corresponding directories are:

* data_models: These are both memory and buffer based. There are helper functions for IO.
* libs: Library functions used by the processing components, not by EF. Not shown on this diagram.
* processing_components: components that can be executed by the Execution Framework.
* workflows: Contains top level workflows

An execution framework can access the processing components via the processing_component_interface which is contained
 in processing_components.external interface.

The data models are implemented as python classes. The functions are stateless so calling the same function with the
same inputs will always return the same value.

Please note that the documentation below has links to the python source. In the end, this is the definitive
documentation.

Functional Model
================

The top-level functional model corresponds to the major SDP pipelines:

.. image:: ./ARL_functional.png
   :scale: 100 %


ARL API
=======

The data structures are operated on by state-less functions as described below.


Data
----

Memory Data models
++++++++++++++++++

.. automodule:: data_models.memory_data_models
   :members:

Buffer Data Models
++++++++++++++++++

.. automodule:: data_models.buffer_data_models
   :members:

Data model persistence
++++++++++++++++++++++

.. automodule:: data_models.data_model_helpers
   :members:



Parameter handling
++++++++++++++++++

.. automodule:: data_models.parameters
   :members:

Polarisation
++++++++++++

.. automodule:: data_models.polarisation
   :members:

The functions are listed next:

Image
-----

Operations
++++++++++

.. automodule:: processing_components.image.operations
   :members:

Gather/Scatter
++++++++++++++

.. automodule:: processing_components.image.gather_scatter
   :members:

Iterators
+++++++++

.. automodule:: libs.image.iterators
   :members:

Deconvolution
+++++++++++++

.. automodule:: processing_components.image.deconvolution
   :members:

Cleaners
++++++++

.. automodule:: libs.image.cleaners
   :members:

Solvers
+++++++

.. automodule:: processing_components.image.solvers
   :members:

Calibration
-----------

Calibration
+++++++++++

.. automodule:: processing_components.calibration.calibration
   :members:

Operations
++++++++++

.. automodule:: processing_components.calibration.operations
   :members:

Solvers
+++++++

.. automodule:: libs.calibration.solvers
   :members:


SkyModel calibration
++++++++++++++++++++

.. automodule:: processing_components.calibration.calskymodel
   :members:

Fourier transform
-----------------

FFT support
+++++++++++

.. automodule:: libs.fourier_transforms.fft_support
   :members:

Convolutional Gridding
++++++++++++++++++++++

.. automodule:: libs.fourier_transforms.convolutional_gridding
   :members:


Imaging
-------

Base
++++

.. automodule:: processing_components.imaging.base
   :members:

Imaging functions
+++++++++++++++++

.. automodule:: processing_components.imaging.imaging_functions
   :members:

Parameters
++++++++++

.. automodule:: libs.imaging.imaging_params
   :members:

Timeslice
+++++++++

.. automodule:: processing_components.imaging.timeslice_single
   :members:


WStack
++++++

.. automodule:: processing_components.imaging.wstack_single
   :members:

Weighting
+++++++++

.. automodule:: processing_components.imaging.weighting
   :members:


Skycomponent
------------

Operations
++++++++++

.. automodule:: processing_components.skycomponent.operations
   :members:

SkyModel
--------

Operations
++++++++++

.. automodule:: processing_components.skymodel.operations
   :members:

Visibility
----------

Base
++++

.. automodule:: processing_components.visibility.base
   :members:

Operations
++++++++++

.. automodule:: processing_components.visibility.operations
   :members:

Coalesce
++++++++

.. automodule:: processing_components.visibility.coalesce
   :members:

Gather/Scatter
++++++++++++++

.. automodule:: processing_components.visibility.gather_scatter
   :members:

Iterators
+++++++++

.. automodule:: processing_components.visibility.iterators
   :members:

Util
----

Array Functions
+++++++++++++++

.. automodule:: libs.util.array_functions
   :members:

Coordinate Support
++++++++++++++++++

.. automodule:: libs.util.coordinate_support
   :members:

Testing Support
+++++++++++++++

.. automodule:: processing_components.simulation.testing_support
   :members:

Execution
---------

Execution (optionally via Dask)
+++++++++++++++++++++++++++++++

.. automodule:: libs.execution_support.arlexecute
   :members:

Dask init
+++++++++

.. automodule:: libs.execution_support.dask_init
   :members:


Pipelines
---------

Pipelines using arlexecute
++++++++++++++++++++++++++

.. automodule:: processing_components.pipelines.pipeline_components
   :members:

Functions
+++++++++

.. automodule:: processing_components.functions.pipeline_functions
   :members:

Support
+++++++

.. automodule:: processing_components.simulation.simulation_components
   :members:

Processing Component Interface
------------------------------

Wrappers
++++++++

.. automodule:: processing_components.processing_component_interface
   :members:

ARL JSON schema
+++++++++++++++

.. automodule:: processing_components.processing_component_interface.arl_json
   :members:

.. automodule:: processing_components.processing_component_interface.arl_json.json_helpers
   :members:


Component wrapper
+++++++++++++++++

.. automodule:: processing_components.processing_component_interface.processing_component_interface
   :members:

Processing component wrapper
++++++++++++++++++++++++++++

.. automodule:: processing_components.processing_component_interface.processing_component_wrappers
   :members:

Execution helpers
+++++++++++++++++

.. automodule:: processing_components.processing_component_interface.execution_helper
   :members:


Unit tests
==========

Unit tests written in standard python style are available.


.. automodule:: tests.libs.test_array_functions
   :members:

.. automodule:: tests.libs.test_convolutional_gridding
   :members:

.. automodule:: tests.libs.test_coordinate_support
   :members:

.. automodule:: tests.libs.test_fft_support
   :members:

.. automodule:: tests.libs.test_msclean
   :members:

.. automodule:: tests.libs.test_parameters
   :members:

.. automodule:: tests.data_models.test_polarisation
   :members:

.. automodule:: tests.libs.test_arlexecute
   :members:

.. automodule:: tests.processing_components.test_calibration_context
   :members:

.. automodule:: tests.processing_components.test_calibration_operations
   :members:

.. automodule:: tests.processing_components.test_calibration_solvers
   :members:

.. automodule:: tests.processing_components.test_calskymodel
   :members:

.. automodule:: tests.processing_components.test_calskymodel_component
   :members:

.. automodule:: tests.processing_components.test_gaintable_iterators
   :members:

.. automodule:: tests.processing_components.test_generic_components
   :members:

.. automodule:: tests.processing_components.test_image_deconvolution
   :members:

.. automodule:: tests.processing_components.test_image_deconvolution_msmfs
   :members:

.. automodule:: tests.processing_components.test_image_gather_scatter
   :members:

.. automodule:: tests.processing_components.test_image_iterators
   :members:

.. automodule:: tests.processing_components.test_image_operations
   :members:

.. automodule:: tests.processing_components.test_image_solvers
   :members:

.. automodule:: tests.processing_components.test_image_solvers_mm
   :members:

.. automodule:: tests.processing_components.test_imaging
   :members:

.. automodule:: tests.processing_components.test_imaging_components
   :members:

.. automodule:: tests.processing_components.test_imaging_deconvolve_component
   :members:

.. automodule:: tests.processing_components.test_imaging_functions
   :members:

.. automodule:: tests.processing_components.test_imaging_params
   :members:

.. automodule:: tests.processing_components.test_pipelines_component
   :members:

.. automodule:: tests.processing_components.test_pipelines_functions
   :members:

.. automodule:: tests.processing_components.test_primary_beams
   :members:

.. automodule:: tests.processing_components.test_quality_assessment
   :members:

.. automodule:: tests.processing_components.test_skycomponent
   :members:

.. automodule:: tests.processing_components.test_skycomponent_insert
   :members:

.. automodule:: tests.processing_components.test_skymodel
   :members:

.. automodule:: tests.processing_components.test_support_components
   :members:

.. automodule:: tests.processing_components.test_testing_support
   :members:

.. automodule:: tests.processing_components.test_visibility_coalesce
   :members:

.. automodule:: tests.processing_components.test_visibility_fitting
   :members:

.. automodule:: tests.processing_components.test_visibility_gather_scatter
   :members:

.. automodule:: tests.processing_components.test_visibility_iterators
   :members:

.. automodule:: tests.processing_components.test_visibility_operations
   :members:

.. automodule:: tests.processing_components.test_visibility_selectors
   :members:

.. automodule:: tests.processing_components.test_weighting
   :members:

