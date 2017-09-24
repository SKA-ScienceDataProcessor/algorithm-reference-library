# simple makefile to simplify repetitive build env management tasks under posix
PYTHON ?= python3
PYLINT ?= pylint3
NOSETESTS ?= nosetests3
FLAKE ?= flake8


all: clean inplace test

clean:
	$(PYTHON) setup.py clean
	rm -rf dist

in: inplace # just a shortcut
inplace:
	$(PYTHON) setup.py install >/dev/null 2>&1 || (echo "'$(PYTHON) setup.py install' failed."; exit -1)

unittest: inplace
	$(PYTHON) -m unittest discover -f --locals -s tests -p "test_*.py"

pytest: inplace
	pytest -x -v tests/

nosetests: inplace
	$(NOSETESTS) -s -v  -e create_low_test_beam -e create_low_test_skycomponents_from_gleam tests/

nosetests-coverage: inplace
	rm -rf coverage .coverage
	$(NOSETESTS) -s -v --with-coverage arl

trailing-spaces:
	find arl -name "*.py" -exec perl -pi -e 's/[ \t]*$$//' {} \;
	find tests -name "*.py" -exec perl -pi -e 's/[ \t]*$$//' {} \;

docs: inplace
	# you must have graphviz installed
	$(MAKE) -C docs dirhtml

code-flake:
	# flake8 ignore long lines and trailing whitespace
	$(FLAKE) --ignore=E501,W293 arl

code-lint:
	$(PYLINT) --extension-pkg-whitelist=numpy \
	  --ignored-classes=astropy.units,astropy.constants,HDUList \
	  -E arl/ tests/

code-analysis: code-flake code-lint

examples: inplace
	$(MAKE) -C examples/arl

notebook:
	DEVICE=`ip link | grep -E " ens| wlan| eth" | grep BROADCAST | tail -1 | cut -d : -f 2  | sed "s/ //"` && \
	IP=`ip a show $${DEVICE} | grep ' inet ' | awk '{print $$2}' | sed 's/\/.*//'` && \
	echo "Launching at IP: $${IP}" && \
	jupyter notebook --no-browser --ip=$${IP} --port=8888 examples/arl/
