.. Documentation master


.. toctree::

Algorithm Reference Library
###########################

The Algorithm Reference Library expresses radio interferometry calibration and imaging algorithms in python and numpy.
The emphasis is on capturing the key algorithms and data models. The interfaces all operate with familiar data
structures such as image, visibility table, gaintable, etc. The python source code is directly accessible from these
documentation pages: see the source link in the top right corner.

To acheive sufficient performance we take a dual pronged approach - using threaded libraries for shared memory
processing, and the `Dask <www.dask.org>`_ library for distributed processing.

.. _getting-started:

Getting Started
***************

.. toctree::
   :maxdepth: 2

   ARL_quickstart
   ARL_directory
   ARL_api
   ARL_otherinfo

* :ref:`genindex`
* :ref:`modindex`

.. _feedback: mailto:realtimcornwell@gmail.com
