#!/bin/bash

# =================================================== #
# Set of comands to Install arl and wrapper into CSD3 #
# =================================================== #

# Load the modules
#module load python-3.5.2-gcc-5.4.0-rdp6q5l
module load python-3.6.1-gcc-5.4.0-23fr5u4
module load git-2.14.1-gcc-5.4.0-acb553e
module load git-lfs-2.3.0-gcc-5.4.0-oktvmkw
module load cfitsio-3.410-gcc-5.4.0-tp3pkyv

# Clone the repository
#git clone https://github.com/SKA-ScienceDataProcessor/algorithm-reference-library/
#cd algorithm-reference-library/

# Get the data
git-lfs pull

# Start the building ARL through building a python virtualenvironment
virtualenv -p `which python3` _build
source _build/bin/activate
pip install --upgrade pip
pip install -U setuptools
pip install coverage numpy
pip install -r requirements.txt 

# Install mpi4py
pip install mpi4py


pip install virtualenvwrapper

echo 'Adding the arl and ffiwrappers path to the virtual environment'
echo '(equivalent to setting up PYTHONPATH environment variable)'
# this updates
# _build/lib/python3.x/site-packages/_virtualenv_path_extensions.pth
source virtualenvwrapper.sh
add2virtualenv $PWD
add2virtualenv $PWD/ffiwrappers/src/

# This is required for some systems (i.e. Jenkins server or macos) others
# detect the python libraries alone and link with correct flags without setting
# up
# the flags explicitely
export LDFLAGS="$(python3-config --ldflags) -lcfitsio"
python setup.py install

# Test the ffiwrappers
export ARLROOT=$PWD

