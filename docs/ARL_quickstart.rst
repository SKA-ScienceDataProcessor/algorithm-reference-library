.. Quick start

Quick start
===========


Installation
++++++++++++

Installation should be straightforward. We recommend the use of virtual environment. A prepackaged python
system such as Anaconda https://www.anaconda.com is usually best as a base.

ARL requires python 3.6 or higher.

# Use git to make a local clone of the Github respository::

   git clone https://github.com/SKA-ScienceDataProcessor/algorithm-reference-library

# Change into that directory::

   cd algorithm-reference-library

# Install required python packages::

   pip install -r requirements.txt

There may be some dependencies that require either conda (or brew install on a mac).

# Setup ARL::

   python setup.py install

# Get the data files form Git LFS::

   git-lfs pull

The README.md file contains much more information about installation.

Running notebooks
+++++++++++++++++

The best way to get familiar with ARL is via jupyter notebooks. For example::

   jupyter-notebook imaging.ipynb

See the jupyter note books below:

.. toctree::
   :maxdepth: 3

   workflows/imaging_serial.rst
   workflows/imaging-fits_arlexecute.rst
   workflows/imaging-wterm_arlexecute.rst
   processing_components/rcal.rst
   workflows/simple-dask_arlexecute.rst
   workflows/imaging-pipelines_arlexecute.rst

In addition, there are other notebooks that are not built as part of this documentation.