.. Documentation master


:index:`Crocodile`
******************

Crocodile is an SKA-SDP project to create reference code in NumPy for radio interferometric aperture calibration and
imaging. The libraries, :doc:`Crocodile<crocodilelib>` and :doc:`Algorithm Reference Library<arllib>`, contain
reference algorithms. In addition to the API's, jupyter http://www.jupyter.org notebooks show how to use the libraries.

The :doc:`Crocodile<crocodilelib>` library represents imaging algorithms implemented in a stripped down form as numpy
operations. Gridding/degridding and FFTs are present in a form suitable for single frequency processing.

The :doc:`Algorithm Reference Library<arllib>` represents calibration and imaging algorithms at a higher level
where the data structures and operations would be familiar to those coming from use of the big radio-astronomical
packages. The functions are close (but not identical) to those in the SDP product tree. The ARL is built using some
capabilities from :doc:`Crocodile<crocodilelib>`, as well as astronomical capabilities from Astropy http://www
.astropy.org. The ARL capabilities are in a form that reflects the complexities of data access and distribution. For
example, in Crocodile gridding is performed on a single frequency. In the ARL, a range of frequencies are treated
together, thus capturing opportunities for optimisation.

For more information on the ARL see :doc:`Algorithm Reference Library Goals and Design<arllib_design>`

All data structures have :doc:`python unit tests<tests>`.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
* :doc:`Glossary<glossary>`

