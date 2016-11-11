.. Documentation master

.. toctree::
   :maxdepth: 4
   :numbered:

   arllib
   arllib_goals
   arllib_bestpractices
   arllib_fourier_processing
   arl/imaging
   tests
   glossary
   refs

Algorithm Reference Library
***************************

* The doc:`Algorithm Reference Library<arllib>` is an SKA-SDP project to create reference code in NumPy for radio
  interferometric aperture calibration and imaging. In addition to the API's, jupyter http://www.jupyter.org
  notebooks show how to use the libraries.
* The :doc:`Algorithm Reference Library<arllib>` represents calibration and imaging algorithms at a high level
  where the data structures and operations would be familiar to those coming from use of the big radio-astronomical
  packages. The functions are close (but not identical) to those in the SDP product tree. The ARL is mostly
  self-contained apart from numpy and some astronomical capabilities from Astropy http://www.astropy.org.
* For more information on the goals of the ARL see :doc:`ARL Goals<arllib_goals>`
* For best practices in development seee :doc:`ARL Best practices<arllib_bestpractices>`
* For the Fourier processing framework see :doc:`ARL Fourier processing<arllib_fourier_processing>`
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
* :doc:`Glossary<glossary>`

