#!/bin/bash

# =========================================================== #
# Set of comands to Install arl and wrapper into MacOS system #
# =========================================================== #


# ########################################################### #
# The following changes are required to run arl and ffiwrappers in Darwin: #
# * Need to change Makefile to point to clang compiler and to link with python library (otherwise Python simbols like Py_Initialize are undefined)
#   export CC=clang 
#   export LDFLAGS="$(python3-config --ldflags) -lcfitsio"
#* ./timg_serial
#dyld: Library not loaded: build/lib.macosx-10.6-intel-3.6/libarlffi.cpython-36m-darwin.so
#  Referenced from: /Users/montsefarreras/Documents/repo/algorithm-reference-library/examples/ffi_demo/timg_serial/./timg_serial
#    Reason: image not found
#    Trace/BPT trap: 5

#    S: name of the library is a relative path, so modify the LD_LIBRARY_PATH does not work, solution: define a fallback path
#    export DYLD_FALLBACK_LIBRARY_PATH=$ARLROOT

#    * arlwrap module not found and cffi module not found:
#    problem is that when calling from C, python ignores the python virtual environment created and tries to find modules in the global python installation, by modifying the PYTHONPATH directory we can force it to look at the right place:

#    export PYTHONPATH=$ARLROOT/:$ARLROOT/ffiwrappers/src/:$ARLROOT/_build/lib/python3.6/site-packages/
# ########################################################### #


# Load the dependency modules
# PYTHON 3.6 +
# GIT 2.10+
# GIT-LFS 
# i.e.:
#module load python-3.5.2-gcc-5.4.0-rdp6q5l
#module load python-3.6.1-gcc-5.4.0-23fr5u4
#module load git-2.14.1-gcc-5.4.0-acb553e
#module load git-lfs-2.3.0-gcc-5.4.0-oktvmkw
#module load cfitsio-3.410-gcc-5.4.0-tp3pkyv

#load MPI module or install i.e. brew install mpich

# ########################################################### #
# If repository is cloned skip this part ....                 #
# Clone the repository
#git clone https://github.com/SKA-ScienceDataProcessor/algorithm-reference-library/
#cd algorithm-reference-library/

# Get the data
#git-lfs pull
# ########################################################### #


# ########################################################### #
# This should be executed from ARLROOT                        #
# i.e. source scripts/build-arl-linux.sh                      #
# ########################################################### #

# Start the building ARL through building a python virtualenvironment
virtualenv -p `which python3` _build
source _build/bin/activate
pip install --upgrade pip
pip install -U setuptools
pip install coverage numpy
pip install -r requirements.txt 
pip install virtualenvwrapper

# Install mpi4py
pip install mpi4py

# Adding the ARL and FFIWRAPPERS path to the virtual environment
# This should be equivalent to setting up the PYTHONPATH environemnt
# variable. (I.e. in MacOS this needs to be done as the 
# add2virtualenv does not seem to work 
echo 'Adding the arl and ffiwrappers path to the virtual environment'
source virtualenvwrapper.sh
add2virtualenv $PWD
add2virtualenv $PWD/ffiwrappers/src/
# For MacOS PYTHONPATH needs to be set to the arlroot, and ffiwrappers plus the modules in 
# the created virtual environmet
export ARLROOT=$PWD
export PYTHONPATH=$ARLROOT/:$ARLROOT/ffiwrappers/src/:$ARLROOT/_build/lib/python3.6/site-packages/

# This is required for some systems (i.e. Jenkins server or macos) others
# detect the python libraries alone and link with correct flags without setting up
# the flags explicitely
export LDFLAGS="$(python3-config --ldflags) -lcfitsio"
python setup.py install

# Test the ffiwrappers
# For MacOS some additional steps are needed:
export CC=clang
# This is a workaround to define where the C extension library is located 
# LD_LIBRARY_FLAG does not work because the build binary points to a 
# relative path, so we define a fallback path to look for libraries
# if they are not found anywhere else.
export DYLD_FALLBACK_LIBRARY_PATH=$ARLROOT
source tests/ffiwrapped/run-tests.sh

#ldd libarlffi.so 
#cd timg_serial/
#make run

# Test the MPI version
pip install pytest

