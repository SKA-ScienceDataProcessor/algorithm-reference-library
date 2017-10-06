
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
* Python package dependencies as defined in the tools/requirements.txt

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
   `pip install -r tools/requirements.txt`

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
  * `tools`: package requirements, and [Docker](https://www.docker.com/) image building recipe

Running Notebooks
-----------------

[Jupyter notebooks](http://jupyter.org/) end with `.ipynb` and can be run as follows from the
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

Using Docker
=============

In the tools/ directory is a [Dockerfile](https://docs.docker.com/engine/reference/builder/)
that enables the notebooks, tests, and lint checks to be run in a container.

Install Docker
--------------
This has been tested with Docker-CE 17.09+ on Ubuntu 16.04. This can be installed by following the instructions [here](https://docs.docker.com/engine/installation/).

It is likely that the Makefile commands will not work on anything other than modern Linux systems (eg: Ubuntu 16.04) as it replies on command line tools to discover the host system IP address.

The project source code directory needs to be checked out where ever the containers are to be run
as the complete project is mounted into the container as a volume at `/arl` .

For example - to launch the notebook server, the general recipe is:
```
docker run --name arl_notebook --hostname arl_notebook --volume /path/to/repo:/arl \
-e IP=my.ip.address --net=host -p 8888:8888 -p 8787:8787 -p 8788:8788 -p 8789:8789 \
-d arl_img
```
After a few seconds, check the logs for the Jupyter URL with:
```
docker logs arl_notebook
```
See the `Makefile` for more examples.


Build Image
-----------
To build the container image `arl_img` required to launch the dockerised notebooks,tests, and lint checks run (from the root directory of this checked out project):
```
make docker_build
```

Run
---
To run the Jupyter notebooks:
```
make docker_notebook
```
Wait for the command to complete, and it will print out the URL with token to use for access to the notebooks (example output):
```
...
Successfully built 8a4a7b55025b
...
docker run --name arl_notebook --hostname arl_notebook --volume $(pwd):/arl -e IP=${IP} \
            --net=host -p 8888:8888 -p 8787:8787 -p 8788:8788 -p 8789:8789 -d arl_img
Launching at IP: 10.128.26.15
da4fcfac9af117c92ac63d4228087a05c5cfbb2fc55b2e281f05ccdbbe3ca0be
sleep 3
docker logs arl_notebook
[I 02:25:37.803 NotebookApp] Writing notebook server cookie secret to /root/.local/share/jupyter/runtime/notebook_cookie_secret
[I 02:25:37.834 NotebookApp] Serving notebooks from local directory: /arl/examples/arl
[I 02:25:37.834 NotebookApp] 0 active kernels
[I 02:25:37.834 NotebookApp] The Jupyter Notebook is running at:
[I 02:25:37.834 NotebookApp] http://10.128.26.15:8888/?token=2c9f8087252ea67b4c09404dc091563b16f154f3906282b7
[I 02:25:37.834 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[C 02:25:37.835 NotebookApp]

    Copy/paste this URL into your browser when you connect for the first time,
    to login with a token:
        http://10.128.26.15:8888/?token=2c9f8087252ea67b4c09404dc091563b16f154f3906282b7
```

To run the tests:
```
make docker_tests
```

To run the lint checks:
```
make docker_lint
```
