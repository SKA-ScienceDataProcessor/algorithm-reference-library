# simple makefile to simplify repetitive build env management tasks under posix
PYTHON ?= python3
PYLINT ?= /usr/bin/pylint
NOSETESTS ?= /usr/bin/nosetests
MAKE_DBG ?= ""
TESTS ?= tests/
FLAKE ?= flake8
NAME = arl
IMG ?= $(NAME)_img
TAG ?= ubuntu18.04
DOCKER_IMAGE = $(IMG):$(TAG)
DOCKERFILE ?= Dockerfile.ubuntu18.04
DOCKER = docker
DOCKER_REPO ?= ""
DOCKER_USER ?= ""
DOCKER_PASSWORD ?= ""
WORKER_MEM ?= 512Mi
WORKER_CPU ?= 500m
WORKER_REPLICAS ?= 1
WORKER_ARL_DATA ?= /arl/data
CURRENT_DIR = $(shell pwd)
JUPYTER_PASSWORD ?= changeme
SERVER_DEVICE ?= $(shell ip link | grep BROADCAST | head -1 | awk '{print $$2}' | sed 's/://')
NFS_SERVER ?= "127.0.0.1"

# define overides for above variables in here
-include PrivateRules.mak

checkvars:
	@echo "Image: $(DOCKER_IMAGE)"
	@echo "Repo: $(DOCKER_REPO)"
	@echo "Host net device: $(SERVER_DEVICE)"
	@echo "Nfs: $(NFS_SERVER)"

all: clean build nosetests

docker_all: clean build docker_build docker_tests

clean:
	$(PYTHON) setup.py clean --all
	rm libarlffi.*.so
	rm -rf dist

in: inplace # just a shortcut
inplace:
	$(PYTHON) setup.py build >/dev/null 2>&1 && $(PYTHON) setup.py install >/dev/null 2>&1 || (echo "'$(PYTHON) setup.py install' failed."; exit -1)

build: in

# clean out the cache before tests are run
cleantests:
	cd tests && rm -rf __pycache__

unittest: cleantests
	$(PYTHON) -m unittest discover -f --locals -s tests -p "test_*.py"

pytest: cleantests
	pytest -x -v $(TESTS)

nosetests: cleantests
	rm -f predict_facet_timeslice_graph_wprojection.png pipelines-timings_*.csv
	$(NOSETESTS) -s -v -e create_low_test_beam -e create_low_test_skycomponents_from_gleam $(TESTS)

nosetests-coverage: inplace cleantests
	rm -rf coverage .coverage
	$(NOSETESTS) -s -v --with-coverage libs

trailing-spaces:
	find libs -name "*.py" -exec perl -pi -e 's/[ \t]*$$//' {} \;
	find processing_components -name "*.py" -exec perl -pi -e 's/[ \t]*$$//' {} \;
	find workflows -name "*.py" -exec perl -pi -e 's/[ \t]*$$//' {} \;

docs: inplace
	# you must have graphviz installed
	$(MAKE) -C docs dirhtml

code-flake:
	# flake8 ignore long lines and trailing whitespace
	$(FLAKE) --ignore=E501,W293,F401 --builtins=ModuleNotFoundError libs

code-lint:
	$(PYLINT) --extension-pkg-whitelist=numpy \
	  --ignored-classes=astropy.units,astropy.constants,HDUList \
	  -E libs/ tests/

code-analysis: code-flake code-lint

examples: inplace
	$(MAKE) -C processing_library/notebooks
	$(MAKE) -C processing_components/notebooks
	$(MAKE) -C workflows/notebooks

notebook:
	DEVICE=`ip link | grep -E " ens| wlan| eth" | grep BROADCAST | tail -1 | cut -d : -f 2  | sed "s/ //"` && \
	IP=`ip a show $${DEVICE} | grep ' inet ' | awk '{print $$2}' | sed 's/\/.*//'` && \
	echo "Launching at IP: $${IP}" && \
	jupyter notebook --no-browser --ip=$${IP} --port=8888 examples/arl/

docker_build:
	$(DOCKER) build -t $(DOCKER_IMAGE) -f $(DOCKERFILE) --build-arg PYTHON=$(PYTHON) .

docker_push: docker_build
	docker tag $(DOCKER_IMAGE) $(DOCKER_REPO)$(DOCKER_IMAGE)
	docker push $(DOCKER_REPO)$(DOCKER_IMAGE)

docker_nfs_arl_data:
	CONTAINER_EXISTS=$$($(DOCKER) ps -aqf ancestor=itsthenetwork/nfs-server-alpine) && \
	if [ -n "$${CONTAINER_EXISTS}" ]; then $(DOCKER) rm -f $${CONTAINER_EXISTS}; fi
	docker run -d --name nfs --privileged -p 2049:2049 \
	-v $(CURRENT_DIR)/:/arl \
	-e SHARED_DIRECTORY=/arl itsthenetwork/nfs-server-alpine:latest

k8s_deploy_scheduler:
	DOCKER_IMAGE=$(DOCKER_REPO)$(DOCKER_IMAGE) \
	 envsubst < k8s/resources/k8s-dask-scheduler-deployment.yml | kubectl apply -f -

k8s_delete_scheduler:
	DOCKER_IMAGE=$(DOCKER_REPO)$(DOCKER_IMAGE) \
	 envsubst < k8s/resources/k8s-dask-scheduler-deployment.yml | kubectl delete -f - || true

k8s_deploy_worker:
	DOCKER_IMAGE=$(DOCKER_REPO)$(DOCKER_IMAGE) \
	WORKER_MEM=$(WORKER_MEM) WORKER_CPU=$(WORKER_CPU) \
	WORKER_REPLICAS=$(WORKER_REPLICAS) \
	WORKER_ARL_DATA=$(WORKER_ARL_DATA) \
	NFS_SERVER=$(NFS_SERVER) \
	 envsubst < k8s/resources/k8s-dask-worker-deployment.yml | kubectl apply -f -

k8s_delete_worker:
	DOCKER_IMAGE=$(DOCKER_REPO)$(DOCKER_IMAGE) \
	WORKER_MEM=$(WORKER_MEM) WORKER_CPU=$(WORKER_CPU) \
	WORKER_REPLICAS=$(WORKER_REPLICAS) \
	WORKER_ARL_DATA=$(WORKER_ARL_DATA) \
	NFS_SERVER=$(NFS_SERVER) \
	 envsubst < k8s/resources/k8s-dask-worker-deployment.yml | kubectl delete -f - || true

k8s_deploy_notebook:
	DOCKER_IMAGE=$(DOCKER_REPO)$(DOCKER_IMAGE) \
	WORKER_ARL_DATA=$(WORKER_ARL_DATA) \
	NFS_SERVER=$(NFS_SERVER) \
	 envsubst < k8s/resources/k8s-dask-notebook-deployment.yml | kubectl apply -f -

k8s_delete_notebook:
	DOCKER_IMAGE=$(DOCKER_REPO)$(DOCKER_IMAGE) \
	WORKER_ARL_DATA=$(WORKER_ARL_DATA) \
	NFS_SERVER=$(NFS_SERVER) \
	 envsubst < k8s/resources/k8s-dask-notebook-deployment.yml | kubectl delete -f - || true

docker_notebook: docker_build
	CTNR=`$(DOCKER) ps -q -f name=$(NAME)_notebook` && \
	if [ -n "$${CTNR}" ]; then $(DOCKER) rm -f $(NAME)_notebook; fi
	DEVICE=`ip link | grep -E " ens| wlan| eth" | grep BROADCAST | tail -1 | cut -d : -f 2  | sed "s/ //"` && \
	IP=`ip a show $${DEVICE} | grep ' inet ' | awk '{print $$2}' | sed 's/\/.*//'` && \
	echo "Launching at IP: $${IP}" && \
	$(DOCKER) run --name $(NAME)_notebook --hostname $(NAME)_notebook --volume $$(pwd)/data:/arl/data -e IP=$${IP} \
            --net=host -p 8888:8888 -p 8787:8787 -p 8788:8788 -p 8789:8789 -d $(DOCKER_IMAGE)
	sleep 3
	$(DOCKER) logs $(NAME)_notebook

k8s_deploy: k8s_deploy_scheduler k8s_deploy_worker k8s_deploy_notebook  

k8s_delete: k8s_delete_notebook k8s_delete_worker k8s_delete_scheduler

docker_test_data:
	CTNR=`$(DOCKER) ps -q -f name=helper` && \
	if [ -n "$${CTNR}" ]; then $(DOCKER) rm -f helper; fi
	docker volume rm -f arl-volume || true
	docker volume create --name arl-volume
	$(DOCKER) run -v arl-volume:/data --name helper busybox true
	$(DOCKER) cp $$(pwd)/data/. helper:/data
	$(DOCKER) rm -f helper

docker_tests: cleantests docker_test_data
	rm -f predict_facet_timeslice_graph_wprojection.png pipelines-timings_*.csv
	CTNR=`$(DOCKER) ps -q -f name=$(NAME)_tests` && \
	if [ -n "$${CTNR}" ]; then $(DOCKER) rm -f $(NAME)_tests; fi
	$(DOCKER) run --rm --name $(NAME)_tests --hostname $(NAME) --volume arl-volume:/arl/data \
		            $(DOCKER_IMAGE) /bin/sh -c "cd /arl && make $(MAKE_DBG) nosetests TESTS=\"${TESTS}\""

docker_pytest: cleantests docker_test_data
	CTNR=`$(DOCKER) ps -q -f name=$(NAME)_tests` && \
	if [ -n "$${CTNR}" ]; then $(DOCKER) rm -f $(NAME)_tests; fi
	$(DOCKER) run --rm --name $(NAME)_tests --hostname $(NAME) --volume $$(pwd)/data:/arl/data \
			    $(DOCKER_IMAGE) /bin/sh -c "cd /arl && make $(MAKE_DBG) pytest"

docker_lint:
	CTNR=`$(DOCKER) ps -q -f name=$(NAME)_lint` && \
	if [ -n "$${CTNR}" ]; then $(DOCKER) rm -f $(NAME)_lint; fi
	$(DOCKER) run --rm --name $(NAME)_lint --hostname $(NAME) \
		            $(DOCKER_IMAGE) /bin/sh -c "cd /arl && make $(MAKE_DBG) code-analysis"

docker_shell:
	CTNR=`$(DOCKER) ps -q -f name=$(NAME)_lint` && \
	if [ -n "$${CTNR}" ]; then $(DOCKER) rm -f $(NAME)_lint; fi
	$(DOCKER) run --rm --name $(NAME)_lint --hostname $(NAME) --volume $$(pwd):/arl \
					-v /etc/passwd:/etc/passwd:ro --user=$$(id -u) \
					-v $${HOME}:$${HOME} -w $${HOME} \
					-e HOME=$${HOME} \
		            --net=host -ti $(DOCKER_IMAGE) sh

launch_dask:
	cd tools && ansible-playbook -i ./inventory ./docker.yml
