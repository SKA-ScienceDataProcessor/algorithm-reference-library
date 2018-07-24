.. toctree::
   :maxdepth: 2

ARL workflows directory
***********************

Workflows created using the ARL processing components and the arlexecute interface.

Calibration
===========

* Calibrate workflow: :py:mod:`workflows.arlexecute.calibration.calibration.calibrate_workflow`

Image
=====

* Generic image iteration: :py:mod:`workflows.arlexecute.image.image_workflows`
* Generic image workflow: :py:mod:`workflows.arlexecute.image.image_workflows.generic_imageworkflow`

Imaging
=======

* Invert: :py:mod:`workflows.arlexecute.imaging.imaging_workflows.invert_workflow`
* Predict: :py:mod:`workflows.arlexecute.imaging.invert_workflows.predict_workflow`
* Deconvolve: :py:mod:`workflows.arlexecute.imaging.invert_workflows.deconvolve_workflow`

Model Partition Calibration
===========================

* Define model partition: :py:mod:`workflows.arlexecute.calibration.modelpartition_workflows.modelpartition_solve`
* Solve model partition: :py:mod:`workflows.arlexecute.calibration.modelpartition_workflows.create_modelpartition`

Pipelines
=========

* Execution framework (an interface to Dask): :py:mod:`workflows.arlexecute.execution_support.arlexecute`
* ICAL: :py:mod:`workflows.arlexecute.pipelines.pipeline_workflows.ical_workflow`
* Continuum imaging: :py:mod:`workflows.arlexecute.pipelines.pipeline_workflows.continuum_imaging_workflow`
* Spectral line imaging: :py:mod:`workflows.arlexecute.pipelines.pipeline_workflows.spectral_line_imaging_workflow`

Simulation
==========

* Testing and simulation support: :py:mod:`workflows.arlexecute.simulation.simulation_workflows.simulate_workflow`

Visibility
==========

* Generic visibility function: :py:mod:`workflows.arlexecute.visibility.generic_blockvisibility_workflow`

External interface
==================

* Calling ARL functions externally: :py:mod:`workflows.arlexecute.processing_component_interface`
