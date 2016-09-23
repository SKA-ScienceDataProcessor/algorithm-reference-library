.. ARL documentation master file

.. toctree::
   :name: mastertoc
   :maxdepth: 2


:index:`Fourier processing`
***************************


Goals
-----

There are many algorithms for imaging, using different approaches to correct for various effects:

+ Simple 2D transforms
+ Faceted transforms
+ W projection
+ W snapshots
+ W slices
+ MFS variants of all the above

Approach
--------

Expressing any one of these is straightforward but combinations can become messy unless a suitable abstraction is found.

The pattern used in these algorithms is abstracted in the following diagram:

.. image:: ./ARL_fourier_processing.png
      :width: 1024px


