
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
   :scale: 50 %


ARL API
=======

The data structures are operated on by state-less functions. The complete set of functions is:

.. image:: ./ARL_all.png
   :width: 1024px

Data
----

Data models
+++++++++++

.. automodule:: arl.data.data_models
   :members:


Parameter handling
++++++++++++++++++

.. automodule:: arl.data.parameters
   :members:

Polarisation
++++++++++++

.. automodule:: arl.data.polarisation
   :members:

Image
-----

Operations
++++++++++

.. automodule:: arl.image.operations
   :members:

Iterators
+++++++++

.. automodule:: arl.image.iterators
   :members:

Deconvolution
+++++++++++++

.. automodule:: arl.image.deconvolution
   :members:

Fourier transform
-----------------

FFT support
+++++++++++

.. automodule:: arl.fourier_transforms.fft_support
   :members:

FTProcessor
+++++++++++

.. automodule:: arl.fourier_transforms.ftprocessor_base
   :members:

.. automodule:: arl.fourier_transforms.ftprocessor_facets
   :members:

.. automodule:: arl.fourier_transforms.ftprocessor_wslice
   :members:

.. automodule:: arl.fourier_transforms.ftprocessor_timeslice
   :members:

.. automodule:: arl.fourier_transforms.ftprocessor_params
   :members:

.. automodule:: arl.fourier_transforms.ftprocessor_iterated
   :members:

Convolutional Gridding
++++++++++++++++++++++

.. automodule:: arl.fourier_transforms.convolutional_gridding
   :members:

Skycomponent
------------

Operations
++++++++++

.. automodule:: arl.skycomponent.operations
   :members:

Visibility
----------

Coalesce
++++++++

.. automodule:: arl.visibility.coalesce
   :members:

Iterators
+++++++++

.. automodule:: arl.visibility.iterators
   :members:

Operations
++++++++++

.. automodule:: arl.visibility.operations
   :members:

Util
----

Array Functions
+++++++++++++++

.. automodule:: arl.util.array_functions
   :members:

Coordinate Support
++++++++++++++++++

.. automodule:: arl.util.coordinate_support
   :members:

Quality assessment
++++++++++++++++++

.. automodule:: arl.util.quality_assessment
   :members:


Testing Support
+++++++++++++++

.. automodule:: arl.util.testing_support
   :members:

Dask
----

Dask Graphs
+++++++++++

.. automodule:: arl.graphs.dask_graphs
   :members:

Generic Dask Graphs
+++++++++++++++++++

.. automodule:: arl.graphs.generic_dask_graphs
   :members:

Dask init
+++++++++

.. automodule:: arl.graphs.dask_init
   :members:


Pipelines
---------

Functions
+++++++++

.. automodule:: arl.pipelines.functions
   :members:

Support
+++++++

.. automodule:: arl.pipelines.support
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

.. automodule:: tests.test_coalesce
   :members:
   :undoc-members:

.. automodule:: tests.test_convolutional_gridding
   :members:
   :undoc-members:

.. automodule:: tests.test_coordinate_support
   :members:
   :undoc-members:

.. automodule:: tests.test_fft_support
   :members:
   :undoc-members:

.. automodule:: tests.test_ftprocessor
   :members:
   :undoc-members:

.. automodule:: tests.test_ftprocessor_params
   :members:
   :undoc-members:

.. automodule:: tests.test_image_deconvolution
   :members:
   :undoc-members:

.. automodule:: tests.test_image_deconvolution_msmfs
   :members:
   :undoc-members:

.. automodule:: tests.test_image_msclean
   :members:
   :undoc-members:

.. automodule:: tests.test_image_iterators
   :members:
   :undoc-members:

.. automodule:: tests.test_image_operations
   :members:
   :undoc-members:

.. automodule:: tests.test_image_solvers
   :members:
   :undoc-members:

.. automodule:: tests.test_parameters
   :members:
   :undoc-members:

.. automodule:: tests.test_pipelines
   :members:
   :undoc-members:

.. automodule:: tests.test_pipelines_dask
   :members:
   :undoc-members:

.. automodule:: tests.test_polarisation
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

.. automodule:: tests.test_visibility_iterators
   :members:
   :undoc-members:

.. automodule:: tests.test_visibility_operations
   :members:
   :undoc-members:

