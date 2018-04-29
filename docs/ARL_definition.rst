
.. toctree::
   :maxdepth: 2

ARL structure
=============

The ARL structure maps that used in the SDP processing architecture:

.. image:: ./SDP_processing_architecture.png
   :scale: 100 %

The corresponding directories are:

* data_models: These are memory-based are mapped to the Buffer by matching classes.
* libs: Library functions used by the processing components. These are not callable directly by the EF.
* processing_components: components that can be executed by the Execution Framework.
* workflows: Contains top level workflows

ARL is composed of data models and functions. The data models are implemented as python classes. The functions are
stateless so calling the same function with the same inputs will always return the same value.

Please note that the documentation below has links to the python source. In the end, this is the definitive
documentation.

Functional Model
================

The top-level functional model corresponds to the major SDP pipelines:

.. image:: ./ARL_functional.png
   :scale: 100 %


ARL API
=======

The data structures are operated on by state-less functions. The complete set of functions is:

.. image:: ./ARL_all.png
   :width: 1024px


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

.. automodule:: processing_components.image.iterators
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

Imaging context
+++++++++++++++

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

.. automodule:: processing_components.util.testing_support
   :members:

Execution
---------

Execution (optionally via Dask)
+++++++++++++++++++++++++++++++

.. automodule:: processing_components.component_support.arlexecute
   :members:

Generic execution
+++++++++++++++++

.. automodule:: processing_components.component_support.generic_components
   :members:

Dask init
+++++++++

.. automodule:: processing_components.component_support.dask_init
   :members:


Pipelines
---------

Pipelines using arlexecute
++++++++++++++++++++++++++

.. automodule:: processing_components.components.pipeline_components
   :members:

Functions
+++++++++

.. automodule:: processing_components.functions.pipeline_functions
   :members:

Support
+++++++

.. automodule:: processing_components.components.support_components
   :members:


Unit tests
**********

Unit tests written in standard python style are available.


.. automodule:: libs.tests.test_array_functions
   :members:

.. automodule:: libs.tests.test_convolutional_gridding
   :members:

.. automodule:: libs.tests.test_coordinate_support
   :members:

.. automodule:: libs.tests.test_fft_support
   :members:

.. automodule:: libs.tests.test_msclean
   :members:

.. automodule:: libs.tests.test_parameters
   :members:

.. automodule:: libs.tests.test_polarisation
   :members:

.. automodule:: processing_components.tests.test_arlexecute
   :members:

.. automodule:: processing_components.tests.test_calibration_context
   :members:

.. automodule:: processing_components.tests.test_calibration_operations
   :members:

.. automodule:: processing_components.tests.test_calibration_solvers
   :members:

.. automodule:: processing_components.tests.test_calskymodel
   :members:

.. automodule:: processing_components.tests.test_calskymodel_component
   :members:

.. automodule:: processing_components.tests.test_gaintable_iterators
   :members:

.. automodule:: processing_components.tests.test_generic_components
   :members:

.. automodule:: processing_components.tests.test_image_deconvolution
   :members:

.. automodule:: processing_components.tests.test_image_deconvolution_msmfs
   :members:

.. automodule:: processing_components.tests.test_image_gather_scatter
   :members:

.. automodule:: processing_components.tests.test_image_iterators
   :members:

.. automodule:: processing_components.tests.test_image_operations
   :members:

.. automodule:: processing_components.tests.test_image_solvers
   :members:

.. automodule:: processing_components.tests.test_image_solvers_mm
   :members:

.. automodule:: processing_components.tests.test_imaging
   :members:

.. automodule:: processing_components.tests.test_imaging_components
   :members:

.. automodule:: processing_components.tests.test_imaging_deconvolve_component
   :members:

.. automodule:: processing_components.tests.test_imaging_functions
   :members:

.. automodule:: processing_components.tests.test_imaging_params
   :members:

.. automodule:: processing_components.tests.test_pipelines_component
   :members:

.. automodule:: processing_components.tests.test_pipelines_functions
   :members:

.. automodule:: processing_components.tests.test_primary_beams
   :members:

.. automodule:: processing_components.tests.test_quality_assessment
   :members:

.. automodule:: processing_components.tests.test_skycomponent
   :members:

.. automodule:: processing_components.tests.test_skycomponent_insert
   :members:

.. automodule:: processing_components.tests.test_skymodel
   :members:

.. automodule:: processing_components.tests.test_support_components
   :members:

.. automodule:: processing_components.tests.test_testing_support
   :members:

.. automodule:: processing_components.tests.test_visibility_coalesce
   :members:

.. automodule:: processing_components.tests.test_visibility_fitting
   :members:

.. automodule:: processing_components.tests.test_visibility_gather_scatter
   :members:

.. automodule:: processing_components.tests.test_visibility_iterators
   :members:

.. automodule:: processing_components.tests.test_visibility_operations
   :members:

.. automodule:: processing_components.tests.test_visibility_selectors
   :members:

.. automodule:: processing_components.tests.test_weighting
   :members:

