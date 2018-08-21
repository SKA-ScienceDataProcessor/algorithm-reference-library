.. toctree::
   :maxdepth: 2

Workflows using arlexecute
**************************

Workflows created using the ARL processing components and the arlexecute interface. These are higher level and fewer
in number than processing components.

Calibration workflows
=====================

* Calibrate workflow: :py:mod:`workflows.arlexecute.calibration.calibration_arlexecute.calibrate_list_arlexecute_workflow`


Imaging workflows
=================

* Invert: :py:mod:`workflows.arlexecute.imaging.imaging_arlexecute.invert_list_arlexecute_workflow`
* Predict: :py:mod:`workflows.arlexecute.imaging.imaging_arlexecute.predict_list_arlexecute_workflow`
* Deconvolve: :py:mod:`workflows.arlexecute.imaging.imaging_arlexecute.deconvolve_list_arlexecute_workflow`

Pipeline workflows
==================

* ICAL: :py:mod:`workflows.arlexecute.pipelines.pipeline_arlexecute.ical_list_arlexecute_workflow`
* Continuum imaging: :py:mod:`workflows.arlexecute.pipelines.pipeline_arlexecute.continuum_imaging_list_arlexecute_workflow`
* Spectral line imaging: :py:mod:`workflows.arlexecute.pipelines.pipeline_arlexecute.spectral_line_imaging_list_arlexecute_workflow`

Simulation workflows
====================

* Testing and simulation support: :py:mod:`workflows.arlexecute.simulation.simulation_arlexecute.simulate_list_arlexecute_workflow`

Execution
=========

* Execution framework (an interface to Dask): :py:mod:`wrappers.arlexecute.execution_support.arlexecute`
