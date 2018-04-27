
.. toctree::
   :maxdepth: 2

ARL definition
**************

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

Data models
+++++++++++

.. automodule:: data_models.memory_data_models
   :members:


Parameter handling
++++++++++++++++++

.. automodule:: data_models.memory_parameters
   :members:

Polarisation
++++++++++++

.. automodule:: data_models.memory_polarisation
   :members:

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

.. automodule:: processing_components.image.cleaners
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

Peeling
+++++++

.. automodule:: processing_components.calibration.peeling
   :members:

Solvers
+++++++

.. automodule:: processing_components.calibration.solvers
   :members:


SkyModel calibration
++++++++++++++++++++

.. automodule:: processing_components.calibration.skymodel_cal
   :members:

Fourier transform
-----------------

FFT support
+++++++++++

.. automodule:: processing_components.fourier_transforms.fft_support
   :members:

Convolutional Gridding
++++++++++++++++++++++

.. automodule:: processing_components.fourier_transforms.convolutional_gridding
   :members:


Imaging
-------

Base
++++

.. automodule:: processing_components.imaging.base
   :members:

Imaging context
+++++++++++++++

.. automodule:: processing_components.imaging.imaging_context
   :members:

Parameters
++++++++++

.. automodule:: processing_components.imaging.params
   :members:

Timeslice
+++++++++

.. automodule:: processing_components.imaging.timeslice
   :members:


WStack
++++++

.. automodule:: processing_components.imaging.wstack
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

Execution support
+++++++++++++++++

.. automodule:: processing_components.component_support.arlexecute
   :members:

Quality assessment
++++++++++++++++++

.. automodule:: processing_components.util.quality_assessment
   :members:

Testing Support
+++++++++++++++

.. automodule:: processing_components.util.testing_support
   :members:

Execution
---------

Execution optionally via Dask
+++++++++++++++++++++++++++++

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

Pipeline Graphs using delayed
+++++++++++++++++++++++++++++

.. automodule:: processing_components.components.pipeline_components
   :members:

Functions
+++++++++

.. automodule:: processing_components.pipelines.functions
   :members:

Support
+++++++

.. automodule:: processing_components.components.support_components
   :members:


Unit tests
**********

Unit tests written in standard python style are available.


.. automodule:: tests.test_array_functions
   :members:
   :undoc-members:

.. automodule:: tests.test_calibration_operations
   :members:
   :undoc-members:

.. automodule:: tests.test_calibration_peeling
   :members:
   :undoc-members:

.. automodule:: tests.test_calibration_solvers
   :members:
   :undoc-members:

.. automodule:: tests.test_convolutional_gridding
   :members:
   :undoc-members:

.. automodule:: tests.test_coordinate_support
   :members:
   :undoc-members:

.. automodule:: tests.test_delayed_support
   :members:
   :undoc-members:

.. automodule:: tests.test_fft_support
   :members:
   :undoc-members:

.. automodule:: tests.test_generic_graph
   :members:
   :undoc-members:

.. automodule:: tests.test_image_deconvolution
   :members:
   :undoc-members:

.. automodule:: tests.test_image_deconvolution_msmfs
   :members:
   :undoc-members:

.. automodule:: tests.test_image_gather_scatter
   :members:
   :undoc-members:

.. automodule:: tests.test_image_iterators
   :members:
   :undoc-members:

.. automodule:: tests.test_image_msclean
   :members:
   :undoc-members:

.. automodule:: tests.test_image_operations
   :members:
   :undoc-members:

.. automodule:: tests.test_image_solvers
   :members:
   :undoc-members:

.. automodule:: tests.test_imaging_delayed
   :members:
   :undoc-members:

.. automodule:: tests.test_imaging_functions
   :members:
   :undoc-members:

.. automodule:: tests.test_imaging_params
   :members:
   :undoc-members:

.. automodule:: tests.test_parameters
   :members:
   :undoc-members:

.. automodule:: tests.test_pipelines_delayed
   :members:
   :undoc-members:

.. automodule:: tests.test_pipelines_functions
   :members:
   :undoc-members:

.. automodule:: tests.test_polarisation
   :members:
   :undoc-members:

.. automodule:: tests.test_primary_beams
   :members:
   :undoc-members:

.. automodule:: tests.test_quality_assessment
   :members:
   :undoc-members:

.. automodule:: tests.test_skycomponent
   :members:
   :undoc-members:

.. automodule:: tests.test_testing_support
   :members:
   :undoc-members:

.. automodule:: tests.test_visibility_coalesce
   :members:
   :undoc-members:

.. automodule:: tests.test_visibility_iterators
   :members:
   :undoc-members:

.. automodule:: tests.test_visibility_operations
   :members:
   :undoc-members:

.. automodule:: tests.test_weighting
   :members:
   :undoc-members:
