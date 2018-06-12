.. toctree::
   :maxdepth: 2

ARL directory
*************

A simple directory to help those familiar with other calibration and imaging packages navigate the Algorithm Reference
Library.

See also the :ref:`genindex` and :ref:`modindex`.

Data containers used by ARL
===========================

See :py:mod:`data_models.memory_data_models` for the following definitions:`

* Telescope Configuration: :py:class:`data_models.memory_data_models.Configuration`
* GainTable for gain solutions (as (e.g. output from solve_gaintable): :py:class:`data_models.memory_data_models.GainTable`
* Image (data and WCS header): :py:class:`data_models.memory_data_models.Image`
* Skycomponent ((data for a point source or a Gaussian source): :py:class:`data_models.memory_data_models.Skycomponent`
* SkyModel (A collection of SkyComponents and Images: :py:class:`data_models.memory_data_models.SkyModel`
* Baseline based visibility tables shape (npol,), length nvis) :py:class:`data_models.memory_data_models.Visibility`
* Antenna-based visibility table, shape (nants, nants, nchan, npol), length ntimes): :py:class:`data_models.memory_data_models.BlockVisibility`

Create empty data set for observation (a-la makeMS)
===================================================

* For Visibility operations: :py:mod:`processing_components.visibility.base.create_visibility`
* For BlockVisibility: :py:mod:`processing_components.visibility.base.create_blockvisibility`

Read existing Measurement Set
=============================

Casacore must be installed:

* Creates a list of Visibilities, one per FIELD_ID: :py:mod:`processing_components.visibility.base.create_visibility_from_ms`

Gridding, degridding, and weighting
===================================

* Convolutional gridding: :py:mod:`processing_components.fourier_transforms.convolutional_gridding.convolutional_grid`
* Convolutional degridding: :py:mod:`processing_components.fourier_transforms.convolutional_gridding.convolutional_degrid`
* Generate gridding kernels (standard prolate spheroidal or W-projection): :py:mod:`processing_components.imaging.params.get_kernel_list`

Visibility weighting and tapering
=================================

* Weighting: :py:mod:`processing_components.fourier_transforms.convolutional_gridding.weight_gridding`
* Gaussian tapering: :py:mod:`processing_components.imaging.weighting.taper_visibility_gaussian`
* Tukey tapering: :py:mod:`processing_components.imaging.weighting.taper_visibility_tukey`

Visibility Predict
==================

* Predict BlockVisibility or Visibility for Skycomponent :py:mod:`processing_components.imaging.base.predict_skycomponent_visibility`
* Predict by de-gridding visibilities :py:mod:`processing_components.imaging.imaging_context.predict_function`

Visibility Invert
=================

* Invert by gridding visibilities :py:mod:`processing_components.imaging.imaging_context.invert_function`

Deconvolution
=============

* Deconvolution: :py:mod:`processing_components.image.deconvolution.deconvolve_cube` wraps:
   * Hogbom Clean: :py:mod:`processing_components.image.cleaners.hogbom`
   * Multi-scale Clean: :py:mod:`processing_components.image.cleaners.msclean`
   * Multi-scale multi-frequency Clean: :py:mod:`processing_components.image.cleaners.msmfsclean`
* Restore: :py:mod:`processing_components.image.deconvolution.restore_cube`

Calibration
===========

* Create empty gain table: :py:mod:`processing_components.calibration.operations.create_gaintable_from_blockvisibility`
* Solve for complex gains: :py:mod:`processing_components.calibration.solvers.solve_gaintable`
* Apply complex gains: :py:mod:`processing_components.calibration.operations.apply_gaintable`

Coordinate transforms
=====================

* Station/baseline (XYZ <-> UVW): :py:mod:`processing_components.util.coordinate_support`
* Source (spherical -> tangent plane): :py:mod:`processing_components.util.coordinate_support`
* Phase rotation: :py:mod:`processing_components.visibility.base.phaserotate_visibility`
* Polarisation basis conversion (Linear <-> Stokes <-> Circular): :py:mod:`processing_components.data.polarisation`

Image
=====

* Import from FITS: :py:mod:`processing_components.image.operations.import_image_from_fits`
* Export from FITS: :py:mod:`processing_components.image.operations.export_image_to_fits`
* Reproject: :py:mod:`processing_components.image.operations.reproject_image`
* Smooth image: :py:mod:`processing_components.image.operations.smooth_image`
* FFT: :py:mod:`processing_components.image.operations.fft_image`
* Remove continuum: :py:mod:`processing_components.image.operations.remove_image`
* Convert polarisation:
   * From Stokes To Polarisation: :py:mod:`processing_components.image.operations.convert_stokes_to_polimage`
   * From Polarisation to Stokes: :py:mod:`processing_components.image.operations.convert_polimage_to_stokes`

Visibility
==========

* Coalesce/decoalesce (i.e. BDA) :py:mod:`processing_components.visibility.coalesce`
* Append/sum/divide/QA: :py:mod:`processing_components.visibility.operations`
* Remove continuum: :py:mod:`processing_components.visibility.operations.remove_continuum_blockvisibility`
* Integrate across channels: :py:mod:`processing_components.visibility.operations.integrate_visibility_by_channel`

Pipelines
=========

* Execution framework (an interface to Dask): :py:mod:`processing_components.component_support.arlexecute`
* Prediction and inversion of visibility data: :py:mod:`processing_components.pipelines.pipeline_components`
* Generic image or visibility unary operations::py:mod:`processing_components.component_suppport.generic_components`
* Testing and simulation support: :py:mod:`processing_components.util.testing_support`

Workflows
=========

* ARL component wrappers: :py:mod:`workflows.wrappers`
