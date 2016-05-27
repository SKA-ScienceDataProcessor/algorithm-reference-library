
Crocodile -- Interferometry Imaging Algorithm Reference Library
===============================================================

This is a project to create a reference code in NumPy for somewhat
simplified aperture synthesis imaging.

Warning: The current code is an experimental proof-of-concept. More
here soon.

Motivation
----------

In many software packages, the only function specification is the
application code itself. Although the underlying algorithm may be
documented (e.g. published), the implementation tends to diverge
overtime, making this method of documentation less effective.

The Crocodile algorithm reference library is designed to present
imaging algorithms in a simple Python-based form. This is so that the
implemented functions can be seen and understood without resorting to
interpreting source code shaped by real-world concerns such as
optimisations.

Requirements
------------

This library is built using Python 3.0. We use the following libraries:

  * `jupyter` - for example notebooks
  * `numpy` - for calculations
  * `matplotlib` - for visualisation
  * `pyfits` - for reading reference data

You will have to install these dependencies, either manually using
your package manager of choice or using `pip`:

     $ pip install -r requirements.txt

Acquiring data
--------------

The `data` directory should hold various test data that can be used to
test the functions of this library. However, this is not currently
included in the repository, and must be downloaded and extracted
separately.

You can either have this done automatically:

    $ make -C data

or extract the package found at
http://www.mrao.cam.ac.uk/~pw410/crocodile/data.tar.gz manually into
the `data` folder.

Orientation
-----------

The content of this project is meant for learning and experimentation,
not usage. If you are here to learn about the process of imaging, here
is a quick guide to the project:

  * `crocodile`: The main Python source code
  * `examples`: Usage examples, mainly using Jupyter notebooks.
  * `docs`: Complete documentation. Includes non-interactive output of examples.
  * `data`: Data used

Running Notebooks
-----------------

Jupyter notebooks end with `.ipynb` and can be run as follows from the
command line:

     $ jupyter notebook examples/notebooks/wkernel.ipynb

Building documentation
----------------------

For building the documentation you will need Sphinx as well as
Pandoc. This will extract docstrings from the crocodile source code,
evaluate all notebooks and compose the result to form the
documentation package.

You can build it as follows:

    $ make -C docs [format]

Emit [format] to view a list of documentation formats that Sphinx can
generate for you.
