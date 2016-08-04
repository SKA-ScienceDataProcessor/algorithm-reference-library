.. Documentation master

:index:`Algorithm Reference Library and Crocodile Documentation`
****************************************************************

The libraries documented here contain reference algorithms for radio interferometric calibration and imaging. As well as
API documentation, jupyter http://www.jupyter.org notebooks show how to use the libraries.

The :doc:`Crocodile<crocodile>` library represents imaging algorithms implemented in a stripped down form as numpy
operations. Gridding/degridding and FFTs are present in a form suitable for single frequency processing.


The :doc:`Algorithm Reference Library<arl>` represents calibration and imaging algorithms at a slightly higher level
where the data structures and operations would be familiar to those coming from use of the big radio-astronomical
packages. The functions are close (but not identical) to those in the SDP product tree. The ARL is built using some
capabilities from :doc:`Crocodile<crocodile>`, as well as astronomical capabilities from Astropy http://www.astropy.org.



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
* :doc:`Glossary<glossary>`

