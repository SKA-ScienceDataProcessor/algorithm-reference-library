#!/bin/bash
# Creates a virtualenv and applies (so far) a patch to SciPy 1.0.0 in order to
# deal with _minpack not being thread-safe, breaking multithreaded applications.


REALPATH=$(realpath $0)

ARLROOT=$(dirname $(dirname $REALPATH))

cd $ARLROOT/tools

VENV=$ARLROOT/venv-patched

if [[ -d $VENV ]]; then
	echo "Found virtualenv (or conflicting directory) in $VENV"
else
	echo "Did not find a virtualenv in $VENV. Creating..."
	virtualenv $VENV
	source $VENV/bin/activate

	PKGDIR=$VIRTUAL_ENV/lib/python3.6/site-packages

	echo "Installing requirements.txt..."
	pip install -r $ARLROOT/requirements.txt

	# Check for patch requirements
	# Scipy 1.0.0 has a thread safety issue in _minpack. Test for this version,
	# and apply patch @ https://github.com/scipy/scipy/pull/7999.
	# N.B. Worked around in 1.1.0
	SCIPYVER=$(pip list 2>/dev/null |sed -En 's/scipy \(([0-9\.]*)\)/\1/p')
	if [ $SCIPYVER == "1.0.0" ]; then
		echo "Found scipy version: $SCIPYVER. Applying thread-safety patch for minpack."

		curl -o minpack.patch https://patch-diff.githubusercontent.com/raw/scipy/scipy/pull/7999.patch

		pushd $PKGDIR
		patch -p1 -u < $ARLROOT/tools/minpack.patch
		popd

		rm minpack.patch
	fi

fi
