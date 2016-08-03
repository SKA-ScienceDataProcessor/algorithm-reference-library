.. ARL documentation master file

Algorithm Reference Library
***************************

This is the prime interface for the Algorithm Reference Library. It is designed by the following principles:

+ Data are held in Classes
+ The data members of the classes are directly accessible by name e.g. .data, .name. .phasecentre
+ The data Classes correspond to familiar concepts in radio astronomy packages e.g. visibility, gaintable, image
+ All data manipulation is performed by stateless functions
+ All function parameters are passed by the kwargs mechanism

.. toctree::
   :name: mastertoc
   :maxdepth: 2

ARL-based Notebooks
-------------------

.. toctree::
   :name: mastertoc
   :maxdepth: 2

   arl/imaging

ARL API
-------

.. toctree::
   :name: mastertoc
   :maxdepth: 2


Calibration
+++++++++++

.. automodule:: arl.calibration
   :members:

Clean
+++++

.. automodule:: arl.clean
   :members:

Configuration
+++++++++++++

.. automodule:: arl.configuration
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


Crocodile Documentation
=======================

Crocodile is a collection of functions designed for low+level operations in radio+interferometric imaging.

.. toctree::
   :name: mastertoc
   :maxdepth: 2

Crocodile+based Notebooks
=========================

.. toctree::
   :name: mastertoc
   :maxdepth: 2

   notebooks/coordinates
   notebooks/wkernel
   notebooks/wimaging
   anna/Simulate_uvw
   anna/MakeDtyImg
   anna/MakeDtyImg_variableWsupport_A
   anna/PSWF_Calculation
   ben/read_oskar_vis
   ben/simple_dirty_image
   ben/simple_dirty_image_sphfn

Crocodile API
=============

Simulate
++++++++

.. automodule:: crocodile.simulate
   :members:

Synthesis
+++++++++

.. automodule:: crocodile.synthesis
   :members:

Clean
++++++++

.. automodule:: crocodile.clean
   :members:

Multiscale Clean
++++++++++++++++

.. automodule:: crocodile.msclean
  :members:
