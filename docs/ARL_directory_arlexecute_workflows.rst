.. toctree::
   :maxdepth: 2

Workflows using arlexecute
**************************

Workflows created using the ARL processing components and the arlexecute interface. These are higher level and fewer
in number than processing components.

Calibration workflows
=====================

* Calibrate workflow: :py:mod:`workflows.arlexecute.calibration.calibration_workflows.calibrate_list_arlexecute_workflow`

Model Partition Calibration workflows
=====================================

* Define model partition: :py:mod:`workflows.arlexecute.calibration.modelpartition_workflows.create_modelpartition_list_arlexecute_workflow`
* Solve model partition: :py:mod:`workflows.arlexecute.calibration.modelpartition_workflows.solve_modelpartition_list_arlexecute_workflow`

Image workflows
===============

* Generic image workflow: :py:mod:`workflows.arlexecute.image.image_workflows.generic_image_arlexecute_workflow`
* Generic image iteration workflow: :py:mod:`workflows.arlexecute.image.image_workflows.generic_image_iterator_arlexecute_workflow`

Imaging workflows
=================

* Invert: :py:mod:`workflows.arlexecute.imaging.imaging_workflows.invert_list_arlexecute_workflow`
* Predict: :py:mod:`workflows.arlexecute.imaging.imaging_workflows.predict_list_arlexecute_workflow`
* Deconvolve: :py:mod:`workflows.arlexecute.imaging.imaging_workflows.deconvolve_list_arlexecute_workflow`

Pipeline workflows
==================

* ICAL: :py:mod:`workflows.arlexecute.pipelines.pipeline_workflows.ical_list_arlexecute_workflow`
* Continuum imaging: :py:mod:`workflows.arlexecute.pipelines.pipeline_workflows.continuum_imaging_list_arlexecute_workflow`
* Spectral line imaging: :py:mod:`workflows.arlexecute.pipelines.pipeline_workflows.spectral_line_imaging_list_arlexecute_workflow`

Simulation workflows
====================

* Testing and simulation support: :py:mod:`workflows.arlexecute.simulation.simulation_workflows.simulate_list_arlexecute_workflow`

Visibility workflows
====================

* Generic visibility function: :py:mod:`workflows.arlexecute.visibility.visibility_workflows.generic_blockvisibility_arlexecute_workflow`

Execution
=========

* Execution framework (an interface to Dask): :py:mod:`workflows.arlexecute.execution_support.arlexecute`
