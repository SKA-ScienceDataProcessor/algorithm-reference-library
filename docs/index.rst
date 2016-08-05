.. Documentation master


:index:`Crocodile`
******************

Crocodile is a project to create a reference code in NumPy for radio interferometric aperture calibration and imaging.
The libraries documented here, :doc:`Crocodile<crocodilelib>` and :doc:`Algorithm Reference Library<arllib>`, contain
reference algorithms. In addition to the API's, jupyter http://www.jupyter.org notebooks show how to use the
libraries.

The :doc:`Crocodile<crocodilelib>` library represents imaging algorithms implemented in a stripped down form as numpy
operations. Gridding/degridding and FFTs are present in a form suitable for single frequency processing.

The :doc:`Algorithm Reference Library<arllib>` represents calibration and imaging algorithms at a slightly higher level
where the data structures and operations would be familiar to those coming from use of the big radio-astronomical
packages. The functions are close (but not identical) to those in the SDP product tree. The ARL is built using some
capabilities from :doc:`Crocodile<crocodilelib>`, as well as astronomical capabilities from Astropy http://www.astropy
.org.

In addition, there are :doc:`python unit tests<tests>`.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
* :doc:`Glossary<glossary>`

