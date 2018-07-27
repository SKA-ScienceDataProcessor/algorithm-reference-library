.. toctree::
   :maxdepth: 2

Workflows using arlexecute
**************************

Workflows created using the ARL processing components and the arlexecute interface. These are higher level and fewer
in number than processing components.

Calibration workflows
=====================

* Calibrate workflow: :py:mod:`workflows.arlexecute.calibration.calibration_workflows.calibrate_arlexecute`

Model Partition Calibration workflows
=====================================

* Define model partition: :py:mod:`workflows.arlexecute.calibration.modelpartition_workflows.create_modelpartition_arlexecute`
* Solve model partition: :py:mod:`workflows.arlexecute.calibration.modelpartition_workflows.solve_modelpartition_arlexecute`

Image workflows
===============

* Generic image workflow: :py:mod:`workflows.arlexecute.image.image_workflows.generic_image_arlexecute`
* Generic image iteration workflow: :py:mod:`workflows.arlexecute.image.image_workflows.generic_image_iterator_arlexecute`

Imaging workflows
=================

* Invert: :py:mod:`workflows.arlexecute.imaging.imaging_workflows.invert_arlexecute`
* Predict: :py:mod:`workflows.arlexecute.imaging.imaging_workflows.predict_arlexecute`
* Deconvolve: :py:mod:`workflows.arlexecute.imaging.imaging_workflows.deconvolve_arlexecute`

Pipeline workflows
==================

* ICAL: :py:mod:`workflows.arlexecute.pipelines.pipeline_workflows.ical_arlexecute`
* Continuum imaging: :py:mod:`workflows.arlexecute.pipelines.pipeline_workflows.continuum_imaging_arlexecute`
* Spectral line imaging: :py:mod:`workflows.arlexecute.pipelines.pipeline_workflows.spectral_line_imaging_arlexecute`

Simulation workflows
====================

* Testing and simulation support: :py:mod:`workflows.arlexecute.simulation.simulation_workflows.simulate_arlexecute`

Visibility workflows
====================

* Generic visibility function: :py:mod:`workflows.arlexecute.visibility.visibility_workflows.generic_blockvisibility_arlexecute`

Execution
=========

* Execution framework (an interface to Dask): :py:mod:`workflows.arlexecute.execution_support.arlexecute`

External interface
==================

* Calling ARL functions externally: :py:mod:`workflows.arlexecute.processing_component_interface`