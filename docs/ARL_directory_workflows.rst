.. toctree::
   :maxdepth: 3

Arlexecute workflows directory
******************************

Workflows created using the ARL processing components and the arlexecute interface.

Calibration
===========

* Calibrate workflow: :py:mod:`workflows.arlexecute.calibration.calibration_workflows.calibrate_workflow`
* Define model partition: :py:mod:`workflows.arlexecute.calibration.modelpartition_workflows.create_modelpartition_workflow`
* Solve model partition: :py:mod:`workflows.arlexecute.calibration.modelpartition_workflows.solve_modelpartition_workflow`

Image
=====

* Generic image workflow: :py:mod:`workflows.arlexecute.image.image_workflows.generic_image_workflow`
* Generic image iteration workflow: :py:mod:`workflows.arlexecute.image.image_workflows.generic_image_iterator_workflow`

Imaging
=======

* Invert: :py:mod:`workflows.arlexecute.imaging.imaging_workflows.invert_workflow`
* Predict: :py:mod:`workflows.arlexecute.imaging.imaging_workflows.predict_workflow`
* Deconvolve: :py:mod:`workflows.arlexecute.imaging.imaging_workflows.deconvolve_workflow`

Pipelines
=========

* ICAL: :py:mod:`workflows.arlexecute.pipelines.pipeline_workflows.ical_workflow`
* Continuum imaging: :py:mod:`workflows.arlexecute.pipelines.pipeline_workflows.continuum_imaging_workflow`
* Spectral line imaging: :py:mod:`workflows.arlexecute.pipelines.pipeline_workflows.spectral_line_imaging_workflow`

Simulation
==========

* Testing and simulation support: :py:mod:`workflows.arlexecute.simulation.simulation_workflows.simulate_workflow`

Visibility
==========

* Generic visibility function: :py:mod:`workflows.arlexecute.visibility.visibility_workflows.generic_blockvisibility_workflow`

Execution
=========

* Execution framework (an interface to Dask): :py:mod:`workflows.arlexecute.execution_support.arlexecute`

External interface
==================

* Calling ARL functions externally: :py:mod:`workflows.arlexecute.processing_component_interface`

