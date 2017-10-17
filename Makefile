# simple makefile to simplify repetitive build env management tasks under posix
PYTHON ?= python3.6
PYLINT ?= pylint
NOSETESTS ?= nosetests3
FLAKE ?= flake8
NAME = arl
IMG = $(NAME)_img
TAG = latest
DOCKERFILE = Dockerfile
DOCKER = docker
DOCKER_REPO ?= localhost:5000


all: clean nosetests

clean:
	$(PYTHON) setup.py clean
	rm -rf dist

in: inplace # just a shortcut
inplace:
	$(PYTHON) setup.py install >/dev/null 2>&1 || (echo "'$(PYTHON) setup.py install' failed."; exit -1)

build: in

# clean out the cache before tests are run
cleantests:
	cd tests && rm -rf __pycache__

unittest: cleantests
	$(PYTHON) -m unittest discover -f --locals -s tests -p "test_*.py"

pytest: cleantests
	pytest -x -v tests/

nosetests: cleantests
	rm -f predict_facet_timeslice_graph_wprojection.png pipelines-timings_*.csv
	$(NOSETESTS) -s -v -e create_low_test_beam -e create_low_test_skycomponents_from_gleam tests/

nosetests-coverage: inplace cleantests
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
	$(FLAKE) --ignore=E501,W293,F401 --builtins=ModuleNotFoundError arl

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

docker_build:
	$(DOCKER) build -t $(IMG) -f $(DOCKERFILE) --build-arg PYTHON=$(PYTHON) .

docker_push: docker_build
	docker tag $(IMG) $(DOCKER_REPO)/$(IMG):$(TAG)
	docker push $(DOCKER_REPO)/$(IMG):$(TAG)

docker_notebook: docker_build
	CTNR=`$(DOCKER) ps -q -f name=$(NAME)_notebook` && \
	if [ -n "$${CTNR}" ]; then $(DOCKER) rm -f $(NAME)_notebook; fi
	DEVICE=`ip link | grep -E " ens| wlan| eth" | grep BROADCAST | tail -1 | cut -d : -f 2  | sed "s/ //"` && \
	IP=`ip a show $${DEVICE} | grep ' inet ' | awk '{print $$2}' | sed 's/\/.*//'` && \
	echo "Launching at IP: $${IP}" && \
	$(DOCKER) run --name $(NAME)_notebook --hostname $(NAME)_notebook --volume $$(pwd)/data:/arl/data -e IP=$${IP} \
            --net=host -p 8888:8888 -p 8787:8787 -p 8788:8788 -p 8789:8789 -d $(IMG)
	sleep 3
	$(DOCKER) logs $(NAME)_notebook

docker_tests: docker_build cleantests
	rm -f predict_facet_timeslice_graph_wprojection.png pipelines-timings_*.csv
	CTNR=`$(DOCKER) ps -q -f name=$(NAME)_tests` && \
	if [ -n "$${CTNR}" ]; then $(DOCKER) rm -f $(NAME)_tests; fi
	$(DOCKER) run --rm --name $(NAME)_tests --hostname $(NAME) --volume $$(pwd)/data:/arl/data \
					-v /etc/passwd:/etc/passwd:ro --user=$$(id -u) \
					-v $${HOME}:$${HOME} -w $${HOME} \
					-e HOME=$${HOME} \
		            --net=host -ti $(IMG) /bin/sh -c "cd /arl && make nosetests"

docker_pytest: docker_build cleantests
	CTNR=`$(DOCKER) ps -q -f name=$(NAME)_tests` && \
	if [ -n "$${CTNR}" ]; then $(DOCKER) rm -f $(NAME)_tests; fi
	$(DOCKER) run --rm --name $(NAME)_tests --hostname $(NAME) --volume $$(pwd)/data:/arl/data \
					-v /etc/passwd:/etc/passwd:ro --user=$$(id -u) \
					-v $${HOME}:$${HOME} -w $${HOME} \
					-e HOME=$${HOME} \
			    --net=host -ti $(IMG) /bin/sh -c "cd /arl && make pytest"

docker_lint: docker_build
	CTNR=`$(DOCKER) ps -q -f name=$(NAME)_lint` && \
	if [ -n "$${CTNR}" ]; then $(DOCKER) rm -f $(NAME)_lint; fi
	$(DOCKER) run --rm --name $(NAME)_lint --hostname $(NAME) --volume $$(pwd):/arl \
					-v /etc/passwd:/etc/passwd:ro --user=$$(id -u) \
					-v $${HOME}:$${HOME} -w $${HOME} \
					-e HOME=$${HOME} \
		            --net=host -ti $(IMG) /bin/sh -c "cd /arl && make code-analysis"

launch_dask:
	cd tools && ansible-playbook -i ./inventory ./docker.yml
