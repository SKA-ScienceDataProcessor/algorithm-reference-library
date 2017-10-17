.. toctree::
   :maxdepth: 2

ARL directory
*************

A simple directory to help those familiar with other calibration and imaging packages navigate the Algorithm Reference
Library.

See also the :ref:`genindex` and :ref:`modindex`.


Data containers used by ARL
===========================

See :py:mod:`arl.data.data_models` for the following definitions:`

* Telescope Configuration: :py:class:`arl.data.data_models.Configuration`
* GainTable for gain solutions (as (e.g. output from solve_gaintable): :py:class:`arl.data.data_models.GainTable`
* Image (data and WCS header): :py:class:`arl.data.data_models.Image`
* Skycomponent ((data for a point source or a Gaussian source): :py:class:`arl.data.data_models.Skycomponent`
* Baseline based visibility tables shape (npol,), length nvis) :py:class:`arl.data.data_models.Visibility`
* Antenna-based visibility table, shape (nants, nants, nchan, npol), length ntimes): :py:class:`arl.data.data_models.BlockVisibility`

Create empty data set for observation (a-la makeMS)
===================================================

* For Visibility operations: :py:mod:`arl.visibility.base.create_visibility`
* For BlockVisibility: :py:mod:`arl.visibility.base.create_blockvisibility`

Read existing Measurement Set
=============================

* Creates a list of Visibilities, one per FIELD_ID: :py:mod:`arl.visibility.base.create_visibility_from_ms`

Gridding, degridding, and weighting
===================================

* Convolutional gridding: :py:mod:`arl.fourier_transforms.convolutional_gridding.convolutional_grid`
* Convolutional degridding: :py:mod:`arl.fourier_transforms.convolutional_gridding.convolutional_degrid`
* Weighting: :py:mod:`arl.fourier_transforms.convolutional_gridding.weight_gridding`
* Tapering: TBD

Visibility Predict
==================

* Predict BlockVisibility for Skycomponent :py:mod:`arl.imaging.base.predict_skycomponent_blockvisibility`
* Predict Visibility for Skycomponent: :py:mod:`arl.imaging.base.predict_skycomponent_visibility`
* Predict by de-gridding visibilities:

 * :py:mod:`arl.imaging.base.predict_2d_base`
 * :py:mod:`arl.imaging.base.predict_2d`
 * :py:mod:`arl.imaging.base.predict_wstack`
 * :py:mod:`arl.imaging.base.predict_wprojection`
 * :py:mod:`arl.imaging.base.predict_timeslice`
 * :py:mod:`arl.imaging.base.predict_facets`
 * :py:mod:`arl.imaging.base.predict_facets_wprojection`
 * :py:mod:`arl.imaging.base.predict_facets_wstack`
 * :py:mod:`arl.imaging.base.predict_wprojection_wstack`


Visibility Invert
=================

* Generate gridding kernels (standard prolate spheroidal or W-projection): :py:mod:`arl.imaging.params.get_kernel_list`
* Invert by gridding visibilities: 

 * :py:mod:`arl.imaging.base.invert_2d_base`
 * :py:mod:`arl.imaging.base.invert_2d`
 * :py:mod:`arl.imaging.base.invert_wstack`
 * :py:mod:`arl.imaging.base.invert_wprojection`
 * :py:mod:`arl.imaging.base.invert_timeslice`
 * :py:mod:`arl.imaging.base.invert_facets`
 * :py:mod:`arl.imaging.base.invert_facets_wprojection`
 * :py:mod:`arl.imaging.base.invert_facets_wstack`
 * :py:mod:`arl.imaging.base.invert_wprojection_wstack`

Deconvolution
=============

* Deconvolution: :py:mod:`arl.image.deconvolution.deconvolve_cube` wraps:
   * Hogbom Clean: :py:mod:`arl.image.cleaners.hogbom`
   * Multi-scale Clean: :py:mod:`arl.image.cleaners.msclean`
   * Multi-scale multi-frequency Clean: :py:mod:`arl.image.cleaners.msmfsclean`
* Restore: :py:mod:`arl.image.deconvolution.restore_cube`

Calibration
===========

* Create empty gain table: :py:mod:`arl.calibration.operations.create_gaintable_from_blockvisibility`
* Solve for complex gains: :py:mod:`arl.calibration.solvers.solve_gaintable`
* Apply complex gains: :py:mod:`arl.calibration.operations.apply_gaintable`
* Peel compact sources: :py:mod:`arl.calibration.peeling.peel_skycomponent_blockvisibility`

Coordinate transforms
=====================

* Station/baseline (XYZ <-> UVW): :py:mod:`arl.util.coordinate_support`
* Source (spherical -> tangent plane): :py:mod:`arl.util.coordinate_support`
* Phase rotation: :py:mod:`arl.visibility.base.phaserotate_visibility`
* Polarisation basis conversion (Linear <-> Stokes <-> Circular): :py:mod:`arl.data.polarisation`

Image
=====

* Import from FITS: :py:mod:`arl.image.operations.import_image_from_fits`
* Export from FITS: :py:mod:`arl.image.operations.export_image_to_fits`
* Reproject: :py:mod:`arl.image.operations.reproject_image`
* Smooth image: :py:mod:`arl.image.operations.smooth_image`
* FFT: :py:mod:`arl.image.operations.fft_image`
* Remove continuum: :py:mod:`arl.image.operations.remove_image`
* Convert polarisation:
   * From Stokes To Polarisation: :py:mod:`arl.image.operations.convert_stokes_to_polimage`
   * From Polarisation to Stokes: :py:mod:`arl.image.operations.convert_polimage_to_stokes`

Visibility
==========

* Coalesce/decoalesce (i.e. BDA) :py:mod:`arl.visibility.coalesce`
* Append/sum/divide/QA: :py:mod:`arl.visibility.operations`
* Remove continuum: :py:mod:`arl.visibility.operations.remove_continuum_blockvisibility`
* Integrate across channels: :py:mod:`arl.visibility.operations.integrate_visibility_by_channel`

Graphs:
=======

* Perform various types of prediction and inversion of visibility data: :py:mod:`arl.graphs.graphs`
* Perform generic image or visibility unary operations: :py:mod:`arl.graphs.generic_graphs`
* Support testing and simulations: :py:mod:`arl.util.graph_support`
* The canonical pipelines: py:mod:`arl.pipelines.graphs`