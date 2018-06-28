#!/bin/bash

# =========================================================== #
# Set of comands to Install arl and wrapper into Linux system #
# =========================================================== #

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

# Clone the repository
git clone https://github.com/SKA-ScienceDataProcessor/algorithm-reference-library/
cd algorithm-reference-library/

# Get the data
git-lfs pull

# Start the building ARL through building a python virtualenvironment
virtualenv -p `which python3` _build
source _build/bin/activate
pip install --upgrade pip
pip install -U setuptools
pip install coverage numpy
pip install -r requirements.txt 

# Build the ARL C Wrapper
cd examples/ffi_demo/
pip install virtualenvwrapper
source virtualenvwrapper.sh
add2virtualenv $PWD/../..
add2virtualenv $PWD/src
python setup.py build_ext

# Test it
#ldd libarlffi.so 
#cd timg_serial/
#make run

