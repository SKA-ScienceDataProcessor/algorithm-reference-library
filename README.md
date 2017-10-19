
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
* Python 3.6+ (Python 2.x may work (YMMV) but it will retire on the 12/04/2020 so best to migrate now - https://pythonclock.org/)
* git 2.11.0+
* git-lfs 2.2.1+
* Python package dependencies as defined in the tools/requirements.txt

The Python dependencies will install (amongst other things) Jupyter, numpy, scipy, scikit, and Dask.  Because of this it maybe advantageous to setup a virtualenv to contain the project - [instructions can be found here](http://docs.python-guide.org/en/latest/dev/virtualenvs/).

Note that git-lfs is required for some data files. Complete platform dependent installation [instructions can be found here](https://github.com/git-lfs/git-lfs/wiki/Installation).

The current version of Python 3.6 (3.6.1 on Ubuntu 16.04/17.04) has a problem when running Dask throwing the following error:
```
SystemError: unknown opcode
```
It is possible to down grade to Python 3.5, with the caveat that lint and flake8 checking will complain about ModuleError (3.6+ dependency).


Platform Specific Instructions
------------------------------

Ubuntu 17.04+
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

install the required packages for python3.6:
```
sudo apt-get install python3.6-dev python3-pip python3-tk virtualenv virtualenvwrapper

# note that pip will be install as pip3 and python will be python3
# so be mindful to make the correct substitutions below

# Optional for simple Dask test notebook - simple-dask.ipynb
sudo apt-get install graphviz
```

Optional: for those wishing to use pipenv and virtualenv
```
sudo pip3 install pipenv
virtualenv -p python3.6 /path/to/where/you/want/to/keep/arlvenv

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
  * `tools`: package requirements, and [Docker](https://www.docker.com/) image building recipe

Running Notebooks
-----------------

[Jupyter notebooks](http://jupyter.org/) end with `.ipynb` and can be run as follows from the
command line:

     $ jupyter notebook examples/arl/imaging.ipynb

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

Ubuntu 17.04+
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
This has been tested with Docker-CE 17.09+ on Ubuntu 17.04. This can be installed by following the instructions [here](https://docs.docker.com/engine/installation/).

It is likely that the Makefile commands will not work on anything other than modern Linux systems (eg: Ubuntu 17.04) as it relies on command line tools to discover the host system IP address.

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
To build the container image `arl_img` required to launch the dockerised notebooks,tests, and lint checks run (from the root directory of this checked out project) - pass in the PYTHON variable to specify which build of Python to use - python3 or python3.6:
```
make docker_build PYTHON=python3
```

Then push to a given Docker repository:
```
make docker_push PYTHON=python3 DOCKER_REPO=localhost:5000
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

Dask with Docker Swarm
----------------------

If you have a Docker Swarm cluster then the Dask cluster can be launched as follows:

Assuming that the Docker Image for the ARL has been built and pushed to a repository tag eg:
```
10.128.26.15:5000/arl_img:latest
```
And the Swarm cluster master resides on:
```
10.101.1.23
```
And the contents of arl/data is available on every work at:
```
/home/ubuntu/arldata
```
Then launch the Dask Scheduler, and Workers with the following:
```
docker -H 10.101.1.23 service create --detach=true \
 --constraint 'node.role == manager' \
 --name dask_scheduler --network host --mode=global \
   10.128.26.15:5000/arl_img:latest \
   dask-scheduler --host 0.0.0.0 --bokeh --show

docker -H 10.101.1.23 service create --detach=true \
 --name dask_worker --network host --mode=global \
 --mount type=bind,source=/home/ubuntu/arldata,destination=/arl/data \
   10.128.26.15:5000/arl_img:latest \
   dask-worker --host 0.0.0.0 --bokeh --bokeh-port 8788  --nprocs 4 --nthreads 1 --reconnect 10.101.1.23:8786
```

Now you can point the Dask client at the cluster with:
```
export ARL_DASK_SCHEDULER=10.101.1.23:8786
python examples/performance/pipelines-timings.py 4 4
```
