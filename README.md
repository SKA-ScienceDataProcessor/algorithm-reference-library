
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

The actual executable code may be accessed directly from the documentation.

Installing
----------

The ARL has a few dependencies:
* Python 3.5+ (Python 2.x may work but it will retire on the 12/04/2020 so best to migrate now - https://pythonclock.org/)
* git 2.11.0+
* git-lfs 2.2.1+
* Python package dependencies as defined in the requirements.txt

The Python dependencies will install (amongst other things) Jupyter, numpy, scipy, scikit, and Dask.  Because of this it maybe advantageous to setup a virtualenv to contain the project - [instructions can be found here](http://docs.python-guide.org/en/latest/dev/virtualenvs/).

Note that git-lfs is required for some data files. Complete platform dependent installation [instructions can be found here](https://github.com/git-lfs/git-lfs/wiki/Installation).


Platform Specific Instructions
------------------------------

Ubuntu 16.04+
-------------

install required packages for git-lfs:
```
sudo apt-get install software-properties-common python-software-properties build-essential curl
sudo add-apt-repository ppa:git-core/ppa
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs

# the following will update your ~/.gitconfig with the command filter for lfs
git lfs install
```

install the required packages for python3:
```
sudo apt-get install python3-dev python3-pip python3-tk virtualenv virtualenvwrapper

# note that pip will be install as pip3 and python will be python3
# so be mindful to make the correct substitutions below

# Optional for simple Dask test notebook - simple-dask.ipynb
sudo apt-get install graphviz
```

Optional: for those wishing to use pipenv and virtualenv
```
sudo pip3 install pipenv
virtualenv -p python3 /path/to/where/you/want/to/keep/arlvenv

# activate the virtualenv with:
. /path/to/where/you/want/to/keep/arlvenv/bin/activate

# optionally install the Bokeh server to enable the Dask diagnostics web interface
pip install bokeh

# if you want to use pytest
pip install pytest

# permanently fix up the ARL lib path in the virtualenv with the following:
add2virtualenv /path/to/checked/out/algorithm-reference-library

# this updates arlvenv/lib/python3.x/site-packages/_virtualenv_path_extensions.pth

# to turn off/deactivate the virtualenv with:
deactivate
```

Common Installation Process
---------------------------

* Use git to make a local clone of the Github repository::
   `git clone https://github.com/SKA-ScienceDataProcessor/algorithm-reference-library`

* Change into that directory::
   `cd algorithm-reference-library`

* Install required python package::
   `pip install -r requirements.txt`

* Setup ARL::
   `python setup.py install`

* Get the data files form Git LFS::
   `git-lfs pull`

* you may need to fix up the python search path so that Jupyter can find the arl with something like: `export PYTHONPATH=/path/to/checked/out/algorithm-reference-library`


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

     $ jupyter notebook examples/notebooks/imaging.ipynb

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
generate for you. You probably want dirhtml.


Running Tests
=============

Test and code analysis requires nosetests3 and flake8 to be installed.


Platform Specific Instructions
------------------------------

Ubuntu 16.04+
-------------

install flake8, nose, and pylint:
```
sudo apt-get install flake8 python3-nose pylint3
```

Running the Tests
-----------------

All unit tests can be run with:
```
make unittest
```
or nose:
```
make nosetests
```
or pytest:
```
make pytest
```

Code analysis can be run with:
```
make code-analysis
```
