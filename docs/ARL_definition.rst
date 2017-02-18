
ARL definition
**************

Functional Model
================

The functional model corresponds to the pipelines:

.. image:: ./ARL_functional.png
   :scale: 50 %

Data Models
===========

The data models are:

.. image:: ./ARL_data.png
   :scale: 75 %

.. automodule:: arl.data.data_models
   :members:

Processing Parameters
=====================

All components possess an API which is always of the form::

      def processing_function(idatastruct1, idatastruct2, ..., *kwargs):
         return odatastruct1, odatastruct2,... other

Processing parameters are passed via the standard Python kwargs approach.

Inside a function, the values are retrieved can be accessed directly from the
kwargs dictionary, or if a default is needed a function can be used::

    log = get_parameter(kwargs, 'log', None)
    vis = get_parameter(kwargs, 'visibility', None)
    sm = get_parameter(kwargs, 'skymodel', None)

Function parameters should obey a consistent naming convention:

=======  =======
Name     Meaning
=======  =======
vis      Name of Visibility
sm       Name of SkyModel
sc       Name of Skycomponent
gt       Name of GainTable
conf     Name of Configuration
im       Name of input image
qa       Name of quality assessment
log      Name of processing log
=======  =======

If a function argument has a better, more descriptive name e.g. normalised_gt, newphasecentre, use it.

Keyword=value pairs should have descriptive names. The names should be lower case with underscores to separate words:

====================    ==================================  ========================================================
Name                    Meaning                             Example
====================    ==================================  ========================================================
loop_gain               Clean loop gain                     0.1
niter                   Number of iterations                10000
eps                     Fractional tolerance                1e-6
threshold               Absolute threshold                  0.001
fractional_threshold    Threshold as fraction of e.g. peak  0.1
G_solution_interval     Solution interval for G term        100
phaseonly               Do phase-only solutions             True
phasecentre             Phase centre (usually as SkyCoord)  SkyCoord("-1.0d", "37.0d", frame='icrs', equinox=2000.0)
spectral_mode           Visibility processing mode          'mfs' or 'channel'
====================    ==================================  ========================================================

ARL API
=======

The data structures are operated on by state-less components. The complete set of components is:

.. image:: ./ARL_all.png
   :width: 1024px

Data
----

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

Deconvolution
+++++++++++++

.. automodule:: arl.image.deconvolution
   :members:

Operations
++++++++++

.. automodule:: arl.image.operations
   :members:

Iterators
+++++++++

.. automodule:: arl.image.iterators
   :members:

Fourier transform
-----------------

FFT support
+++++++++++

.. automodule:: arl.fourier_transforms.fft_support
   :members:

FTProcessor
+++++++++++

.. automodule:: arl.fourier_transforms.ftprocessor
   :members:

Convolutional Gridding
++++++++++++++++++++++

.. automodule:: arl.fourier_transforms.convolutional_gridding
   :members:

SkyModel
--------

Operations
++++++++++

.. automodule:: arl.skymodel.operations
   :members:

Solvers
+++++++

.. automodule:: arl.skymodel.solvers
   :members:

Visibility
----------

Calibration
+++++++++++

.. automodule:: arl.visibility.calibration
   :members:

Iterators
+++++++++

.. automodule:: arl.visibility.iterators
   :members:

Operations
++++++++++

.. automodule:: arl.visibility.operations
   :members:

Coordinate Support
------------------


Internal Routines
-----------------

.. automodule:: arl.core.c
   :members:

Definition of coordinate systems
++++++++++++++++++++++++++++++++

From http://casa.nrao.edu/Memos/CoordConvention.pdf :

UVW is a right-handed coordinate system, with W pointing towards the
source, and a baseline convention of :math:`ant2 - ant1` where
:math:`index(ant1) < index(ant2)`.  Consider an XYZ Celestial
coordinate system centered at the location of the interferometer, with
:math:`X` towards the East, :math:`Z` towards the NCP and :math:`Y` to
complete a right-handed system. The UVW coordinate system is then
defined by the hour-angle and declination of the phase-reference
direction such that

1. when the direction of observation is the NCP (`ha=0,dec=90`),
   the UVW coordinates are aligned with XYZ,

2. V, W and the NCP are always on a Great circle,

3. when W is on the local meridian, U points East

4. when the direction of observation is at zero declination, an
   hour-angle of -6 hours makes W point due East.

The :math:`(l,m,n)` coordinates are parallel to :math:`(u,v,w)` such
that :math:`l` increases with Right-Ascension (or increasing longitude
coordinate), :math:`m` increases with Declination, and :math:`n` is
towards the source. With this convention, images will have Right
Ascension increasing from Right to Left, and Declination increasing
from Bottom to Top.

Util
----

Coordinate Support
++++++++++++++++++

.. automodule:: arl.util.coordinate_support
   :members:

Testing Support
+++++++++++++++

.. automodule:: arl.util.testing_support
   :members:

Quality Assessment
++++++++++++++++++

.. automodule:: arl.util.quality_assessment
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

.. automodule:: tests.test_convolutional_gridding
   :members:
   :undoc-members:

.. automodule:: tests.test_coordinate_support
   :members:
   :undoc-members:

.. automodule:: tests.test_datamodels
   :members:
   :undoc-members:

.. automodule:: tests.test_fft_support
   :members:
   :undoc-members:

.. automodule:: tests.test_ftprocessor
   :members:
   :undoc-members:

.. automodule:: tests.test_image_deconvolution
   :members:
   :undoc-members:

.. automodule:: tests.test_image_iterators
   :members:
   :undoc-members:

.. automodule:: tests.test_image_operations
   :members:
   :undoc-members:

.. automodule:: tests.test_parameters
   :members:
   :undoc-members:

.. automodule:: tests.test_pipelines
   :members:
   :undoc-members:

.. automodule:: tests.test_quality_assessment
   :members:
   :undoc-members:

.. automodule:: tests.test_visibility_calibration
   :members:
   :undoc-members:

.. automodule:: tests.test_visibility_operations
   :members:
   :undoc-members:

