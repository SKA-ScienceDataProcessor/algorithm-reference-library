.. ARL documentation master file


:index:`Algorithm Reference Library`
************************************

The Algorithm Reference Library is designed by the following principles:

+ Data are held in Classes
+ The data members of the classes are directly accessible by name e.g. .data, .name. .phasecentre
+ The data Classes correspond to familiar concepts in radio astronomy packages e.g. visibility, gaintable, image
+ There are no methods attached to the data Classes apart from variant constructors as needed.
+ All data manipulation is performed by stateless functions
+ All function parameters are passed by the kwargs mechanism

.. toctree::
   :name: mastertoc
   :maxdepth: 2

:index:`ARL-based Notebooks`
----------------------------

.. toctree::
   :name: mastertoc
   :maxdepth: 2

   arl/imaging

:index:`ARL API`
----------------

Calibration
+++++++++++

.. automodule:: arl.calibration
   :members:

Configuration
+++++++++++++

.. automodule:: arl.configuration
   :members:

Deconvolve
++++++++++

.. automodule:: arl.deconvolve
   :members:

Gaintable
+++++++++

.. automodule:: arl.gaintable
   :members:

Image
+++++

.. automodule:: arl.image
   :members:

Imaging
+++++++

.. automodule:: arl.imaging
   :members:

Polarisation
++++++++++++

.. automodule:: arl.polarisation
   :members:

SkyComponent
++++++++++++

.. automodule:: arl.skycomponent
   :members:

SkyModel
++++++++

.. automodule:: arl.skymodel
   :members:

Visibility
++++++++++

.. automodule:: arl.visibility
   :members:
