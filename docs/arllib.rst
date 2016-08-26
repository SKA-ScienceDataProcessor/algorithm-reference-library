.. ARL documentation master file


:index:`Algorithm Reference Library`
************************************

The Algorithm Reference Library is used to capture radio interferometry calibration and imaging algorithms in a
reference form for use by SDP contractors. The interfaces all operate with familiar data structures such as image,
visibility table, gaintable, etc.

See also :doc:`Algorithm Reference Library Goals<arllib_goals>`


.. toctree::
   :name: mastertoc
   :maxdepth: 2

:index:`ARL-based Notebooks`
----------------------------

.. toctree::
   :name: mastertoc
   :maxdepth: 2

   Imaging Demonstration<arl/imaging>

:index:`Data Models`
--------------------

The data models are:

.. image:: ./ARL_data.png

.. automodule:: arl.data_models
   :members:


:index:`ARL API`
----------------


Data structures operated on by state-less components as follows:

.. image:: ./ARL_components.png
      :width: 1024px

All components possess an API which is always of the form::


      def processing_function(idatastruct1, idatastruct2, ..., processingparameters):
         return odatastruct1, odatastruct2,... other


Define Visibility
+++++++++++++++++

.. automodule:: arl.define_visibility
   :members:

Simulate Visibility
+++++++++++++++++++

.. automodule:: arl.simulate_visibility
   :members:

Calibrate Visibility
++++++++++++++++++++

.. automodule:: arl.calibrate_visibility
   :members:

Fourier Transform
+++++++++++++++++

.. automodule:: arl.fourier_transform
   :members:

Deconvolve Image
++++++++++++++++

.. automodule:: arl.deconvolve_image
   :members:

Define Image
++++++++++++

.. automodule:: arl.define_image
   :members:

Polarisation
++++++++++++

.. automodule:: arl.polarisation
   :members:

Define SkyModel
+++++++++++++++

.. automodule:: arl.define_skymodel
   :members:

Pipelines
+++++++++

.. automodule:: arl.pipelines
   :members:

Assess Quality
++++++++++++++

.. automodule:: arl.assess_quality
   :members:


