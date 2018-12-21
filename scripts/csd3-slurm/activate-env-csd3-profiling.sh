#!/bin/bash

# =================================================== #
# Loads environment to run ARL from SLURM in CSD3     #
# =================================================== #

# Load the modules
echo "Loading modules"
#module load python-3.5.2-gcc-5.4.0-rdp6q5l
module load python-3.6.1-gcc-5.4.0-23fr5u4
echo module load python-3.6.1-gcc-5.4.0-23fr5u4
#module load git-2.14.1-gcc-5.4.0-acb553e
#module load git-lfs-2.3.0-gcc-5.4.0-oktvmkw
module load cfitsio-3.410-gcc-5.4.0-tp3pkyv
echo module load cfitsio-3.410-gcc-5.4.0-tp3pkyv
module load intel/bundles/complib/2017.4
echo module load intel/bundles/complib/2017.4
module load use.own
echo "Module path:"
echo $MODULEPATH
echo "Loading Extrae"
export MODULEPATH=$MODULEPATH:$HOME/privatemodules
module load EXTRAE/impi/3.6.1
export EXTRAE_HOME=/home/mf582/scratch/Extrae/extrae-3.6.1-install/impi/

## ARLROOT should be defined already
## export ARLROOT=/home/mf582/scratch/ARL/algorithm-reference-library

#module load python-3.6.2-intel-17.0.4-lv2lxsb 
#module load extrae/3.6.1/cummulus
#export EXTRAE_HOME=/usr/local/software/extrae/

# Start the building ARL through building a python virtualenvironment
#virtualenv -p `which python3` _build
source $ARLROOT/_build/bin/activate
echo "Activating virtual environment"
echo source $ARLROOT/_build/bin/activate

# _build/lib/python3.x/site-packages/_virtualenv_path_extensions.pth
#source virtualenvwrapper.sh
#add2virtualenv $PWD
#add2virtualenv $PWD/ffiwrappers/src/

# This is required for some systems (i.e. Jenkins server or macos) others
# detect the python libraries alone and link with correct flags without setting
# up
# the flags explicitely
export LDFLAGS="$(python3-config --ldflags) -lcfitsio"


