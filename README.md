
Algorithm Reference Library
===========================

This is a project to create a reference code in NumPy for aperture synthesis imaging.

Motivation
----------

In many software packages, the only function specification is the
application code itself. Although the underlying algorithm may be
documented (e.g. published), the implementation tends to diverge
overtime, making this method of documentation less effective.

The Algorithm Reference Library is designed to present calibration and
imaging algorithms in a simple Python-based form. This is so that the
implemented functions can be seen and understood without resorting to
interpreting source code shaped by real-world concerns such as
optimisations.

Installing
----------

# Install git and git-lfs if not already installed. Note that git-lfs is required for some
data files

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

Orientation
-----------

The prime focus of the ARL is on learning and experimentation,
not usage. If you are here to learn about the process of imaging, here
is a quick guide to the project:

  * `arl`: The main Python source code
  * `examples`: Usage examples, mainly using Jupyter notebooks.
  * `tests`: Unit and regression tests
  * `docs`: Complete documentation. Includes non-interactive output of examples.
  * `data`: Data used

Running Notebooks
-----------------

Jupyter notebooks end with `.ipynb` and can be run as follows from the
command line:

     $ jupyter notebook examples/notebooks/wkernel.ipynb

Building documentation
----------------------

The last build documentation is at:

    http://www.mrao.cam.ac.uk/projects/jenkins/algorithm-reference-library/docs/build/html/index.html
    
For building the documentation you will need Sphinx as well as
Pandoc. This will extract docstrings from the crocodile source code,
evaluate all notebooks and compose the result to form the
documentation package.

You can build it as follows:

    $ make -C docs [format]

Omit [format] to view a list of documentation formats that Sphinx can
generate for you.
