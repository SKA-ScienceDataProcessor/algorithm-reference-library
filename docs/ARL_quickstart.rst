.. Definition

Installation
============

# Use git to make a local clone of the Github respository::

   git clone https://github.com/SKA-ScienceDataProcessor/algorithm-reference-library

# Change into that directory::

   cd algorithm-reference-library

# Install required python package::

   pip install -r requirements.txt

# Setup ARL::

   python setup.py install

# Get the data files form Git LFS::

   git-lfs pull

Quick start
===========

The best way to run ARL is via jupyter notebooks. For example::

   jupyter-notebook imaging.ipynb

See the jupyter note books below:

.. toctree::
   :maxdepth: 3

   arl/imaging
   arl/imaging-fits
   arl/imaging-wterm
   arl/imaging-mfs
   arl/rcal
   arl/peeling
   arl/imaging-dask
   arl/imaging-spectral

In addition, there are other notebooks in examples/arl that are not built as part of this documentation. See e.g.
examples/arl/imaging-low.ipynb, examples/arl/imaging-coalesce.ipynb
