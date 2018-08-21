
.. toctree::
   :maxdepth: 2

ARL API
*******

The ARL API consists of data models expressed as python classes, and state-free functions.

The best way to learn the API is via the directory, Jupyter notebooks and the test programs.

.. toctree::
   :maxdepth: 2

Data models
===========

Memory Data models
------------------

.. automodule:: data_models.memory_data_models
   :members:

Buffer Data Models
------------------

.. automodule:: data_models.buffer_data_models
   :members:

Data model persistence
----------------------

.. automodule:: data_models.data_model_helpers
   :members:

Parameter handling
------------------

.. automodule:: data_models.parameters
   :members:

Polarisation
------------

.. automodule:: data_models.polarisation
   :members:

Processing Library
==================

The following are functions that support the processing components but are not expected to interface to workflows.

Image
-----

Cleaners
++++++++

.. automodule:: processing_library.arrays.cleaners
   :members:

Calibration
-----------

Solvers
+++++++

.. automodule:: processing_library.calibration.solvers
   :members:

Fourier transform
-----------------

FFT support
+++++++++++

.. automodule:: processing_library.fourier_transforms.fft_support
   :members:

Convolutional Gridding
++++++++++++++++++++++

.. automodule:: processing_library.fourier_transforms.convolutional_gridding
   :members:


Imaging
-------

Parameters
++++++++++

.. automodule:: processing_library.imaging.imaging_params
   :members:

Util
----

Array Functions
+++++++++++++++

.. automodule:: processing_library.util.array_functions
   :members:

Coordinate Support
++++++++++++++++++

.. automodule:: processing_library.util.coordinate_support
   :members:
      

.. toctree::
   :maxdepth: 2

Processing Components
=====================

These python functions contain the core algorithms in ARL. They can be executed directly or via workflows.


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

Deconvolution
+++++++++++++

.. automodule:: processing_components.image.deconvolution
   :members:

Solvers
+++++++

.. automodule:: processing_components.image.solvers
   :members:

Iterators
+++++++++

.. automodule:: processing_components.image.iterators
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

Modelpartition calibration
++++++++++++++++++++++++++

.. automodule:: processing_components.calibration.modelpartition
   :members:

Imaging
-------

Base
++++

.. automodule:: processing_components.imaging.base
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

Testing Support
+++++++++++++++

.. automodule:: processing_components.simulation.testing_support
   :members:

Workflows using arlexecute
==========================

These are workflows assembled from processing components functions, using the arlexecute framework. arlexecute wraps
Dask in a convenient way so that the decision to use Dask can be made at run time.

`Dask <http://dask.pydata.org/>`_ is a python-based flexible
parallel computing library for analytic computing. arlexecute.execute (equivalent to dask.delayed) can be used to wrap
functions for deferred execution thus allowing construction of graphs. For example, to build a graph for a
major/minor cycle algorithm::

    arlexecute.set_client(use_dask=True)
    model_imagelist = arlexecute.compute(create_image_from_visibility)(vt, npixel=512, cellsize=0.001, npol=1)
    solution_list = create_solve_image_list(vt, model_imagelist=model_imagelist, psf_list=psf_list,
                                            context='timeslice', algorithm='hogbom',
                                            niter=1000, fractional_threshold=0.1,
                                            threshold=1.0, nmajor=3, gain=0.1)
    solution_list.visualize()

    arlexecute.compute(solution_list, sync=True)

If use_dask is set False all steps are executed immediately.

Construction of the components requires that the number of nodes (e.g. w slices or time-slices) be known at construction,
rather than execution. To counteract this, at run time, a given node should be able to act as a no-op. We use None
to denote a null node.

Calibration
-----------

.. automodule:: workflows.arlexecute.calibration.calibration_arlexecute
   :members:

.. automodule:: workflows.arlexecute.calibration.modelpartition_arlexecute
   :members:



.. automodule:: workflows.arlexecute.imaging.imaging_arlexecute
   :members:

Pipelines
---------

.. automodule:: workflows.arlexecute.pipelines.pipeline_arlexecute
   :members:

Simulation
----------

.. automodule:: workflows.arlexecute.simulation.simulation_arlexecute
   :members:

Execution
---------

Execution (optionally via Dask)
+++++++++++++++++++++++++++++++

.. automodule:: wrappers.arlexecute.execution_support.arlexecute
   :members:

Dask init
+++++++++

.. automodule:: wrappers.arlexecute.execution_support.dask_init
   :members:

Serial workflows
================

Imaging functions
-----------------

.. automodule:: workflows.serial.imaging.imaging_serial
   :members:

Pipeline functions
------------------

.. automodule:: workflows.serial.pipelines.pipeline_serial
   :members:

