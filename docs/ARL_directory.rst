.. toctree::
   :maxdepth: 2

Directory
*********

This directory is designed to to help those familiar with other calibration and imaging packages navigate the Algorithm
Reference Library. Not all functions are listed here. The API contains all functions.

The long form of the name is given for all entries but all function names arre unique so a given function can be
accessed using the very top level import::

   import data_models
   import processing_library
   import processing_components
   import workflows

Functions
=========

Data containers used by ARL
---------------------------

ARL holds data in python Classes. The bulk data is usually kept in a python structured array, and the meta data as
attributes.

See :py:mod:`data_models.memory_data_models` for the following definitions:

* Image (data and WCS header): :py:class:`data_models.memory_data_models.Image`
* Skycomponent (data for a point source or a Gaussian source): :py:class:`data_models.memory_data_models.Skycomponent`
* SkyModel (collection of SkyComponents and Images): :py:class:`data_models.memory_data_models.SkyModel`
* Antenna-based visibility table, shape (nants, nants, nchan, npol), length ntimes): :py:class:`data_models.memory_data_models.BlockVisibility`
* Baseline based visibility tables shape (npol,), length nvis) :py:class:`data_models.memory_data_models.Visibility`
* Telescope Configuration: :py:class:`data_models.memory_data_models.Configuration`
* GainTable for gain solutions (as (e.g. output from solve_gaintable): :py:class:`data_models.memory_data_models.GainTable`

Create empty visibility data set for observation (a-la makeMS)
--------------------------------------------------------------

* For Visibility: :py:func:`processing_components.visibility.create_visibility`
* For BlockVisibility: :py:func:`processing_components.visibility.create_blockvisibility`

Read existing Measurement Set
-----------------------------

Casacore must be installed for MS reading and writing:

* List contents of a MeasurementSet: :py:func:`processing_components.visibility.list_ms`
* Creates a list of Visibilities, one per FIELD_ID and DATA_DESC_ID: :py:func:`processing_components.visibility.create_visibility_from_ms`
* Creates a list of BlockVisibilities, one per FIELD_ID and DATA_DESC_ID: :py:func:`processing_components.visibility.create_blockvisibility_from_ms`

Visibility gridding and degridding
----------------------------------

* Convolutional gridding: :py:func:`processing_components.griddata.grid_visibility_to_griddata`
* Convolutional degridding: :py:func:`processing_components.griddata.degrid_visibility_from_griddata`

Visibility weighting and tapering
---------------------------------

* Weighting: :py:func:`processing_components.imaging.weight_visibility`
* Gaussian tapering: :py:func:`processing_components.imaging.taper_visibility_gaussian`
* Tukey tapering: :py:func:`processing_components.imaging.taper_visibility_tukey`

Visibility predict and invert
-----------------------------

* Predict BlockVisibility or Visibility for Skycomponent :py:func:`processing_components.imaging.predict_skycomponent_visibility`
* Predict by de-gridding visibilities :py:func:`workflows.arlexecute.predict_list_arlexecute_workflow`
* Predict by de-gridding visibilities :py:func:`workflows.serial.predict_list_serial_workflow`
* Invert by gridding visibilities :py:func:`workflows.arlexecute.invert_list_arlexecute_workflow`
* Invert by gridding visibilities :py:func:`workflows.serial.invert_list_serial_workflow`

Deconvolution
-------------

* Deconvolution :py:func:`processing_components.image.deconvolve_cube` wraps:
 * Hogbom Clean: :py:func:`processing_library.arrays.hogbom`
 * Hogbom Complex Clean: :py:func:`processing_library.arrays.hogbom_complex`
 * Multi-scale Clean: :py:func:`processing_library.arrays.msclean`
 * Multi-scale multi-frequency Clean: :py:func:`processing_library.arrays.msmfsclean`


* Restore: :py:func:`processing_components.image.restore_cube`

Calibration
-----------

* Create empty gain table: :py:func:`processing_components.calibration.create_gaintable_from_blockvisibility`
* Solve for complex gains: :py:func:`processing_components.calibration.solve_gaintable`
* Apply complex gains: :py:func:`processing_components.calibration.apply_gaintable`

Coordinate transforms
---------------------

* Phase rotation: :py:func:`processing_components.visibility.phaserotate_visibility`
* Station/baseline (XYZ <-> UVW): :py:mod:`processing_library.util.coordinate_support`
* Source (spherical -> tangent plane): :py:mod:`processing_library.util.coordinate_support`

Image
-----

* Image operations: :py:func:`processing_components.image`
* Import from FITS: :py:func:`processing_components.image.import_image_from_fits`
* Export from FITS: :py:func:`processing_components.image.export_image_to_fits`
* Reproject coordinate system: :py:func:`processing_components.image.reproject_image`
* Smooth image: :py:func:`processing_components.image.smooth_image`
* FFT: :py:func:`processing_library.image.fft_image`
* Remove continuum: :py:func:`processing_components.image.remove_continuum_image`
* Convert polarisation:
 * From Stokes To Polarisation: :py:func:`processing_components.image.convert_stokes_to_polimage`
 * From Polarisation to Stokes: :py:func:`processing_components.image.convert_polimage_to_stokes`

Visibility
----------

* Append/sum/divide/QA: :py:func:`processing_components.visibility`
* Remove continuum: :py:func:`processing_components.visibility.remove_continuum_blockvisibility`
* Integrate across channels: :py:func:`processing_components.visibility.integrate_visibility_by_channel`
* Coalesce (i.e. BDA) :py:func:`processing_components.visibility.coalesce_visibility`
* Decoalesce (i.e. BDA) :py:func:`processing_components.visibility.decoalesce_visibility`

Workflows
=========

Workflows coordinate processing using the data models, processing components, and processing library. These are high
level functions, and are available in arlexecute (i.e. dask) version and sometimes scalar version.

Calibration workflows
---------------------

* Calibrate workflow: :py:func:`workflows.arlexecute.calibrate_list_arlexecute_workflow`
* Calibrate workflow: :py:func:`workflows.serial.calibrate_list_serial_workflow`

Imaging workflows
-----------------

* Invert: :py:func:`workflows.arlexecute.invert_list_arlexecute_workflow`
* Predict: :py:func:`workflows.arlexecute.predict_list_arlexecute_workflow`
* Deconvolve: :py:func:`workflows.arlexecute.deconvolve_list_arlexecute_workflow`
* Invert: :py:func:`workflows.serial.invert_list_serial_workflow`
* Predict: :py:func:`workflows.serial.predict_list_serial_workflow`
* Deconvolve: :py:func:`workflows.serial.deconvolve_list_serial_workflow`

Pipeline workflows
------------------

* ICAL: :py:func:`workflows.arlexecute.ical_list_arlexecute_workflow`
* Continuum imaging: :py:func:`workflows.arlexecute.continuum_imaging_list_arlexecute_workflow`
* Spectral line imaging: :py:func:`workflows.arlexecute.spectral_line_imaging_list_arlexecute_workflow`
* MPCCAL: :py:func:`workflows.arlexecute.mpccal_skymodel_list_arlexecute_workflow`
* ICAL: :py:func:`workflows.serial.ical_list_serial_workflow`
* Continuum imaging: :py:func:`workflows.serial.continuum_imaging_list_serial_workflow`
* Spectral line imaging: :py:func:`workflows.serial.spectral_line_imaging_list_serial_workflow`

Simulation workflows
--------------------

* Testing and simulation support: :py:func:`workflows.arlexecute.simulate_list_arlexecute_workflow`
* Testing and simulation support: :py:func:`workflows.serial.simulate_list_serial_workflow`

Execution
---------

* Execution framework (an interface to Dask): :py:func:`wrappers.arlexecute.execution_support.arlexecute`

Scripts
=======




