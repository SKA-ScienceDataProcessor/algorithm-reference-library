# simple makefile to simplify repetitive build env management tasks under posix
PYTHON ?= python3
PYLINT ?= pylint
NOSETESTS ?= nosetests
MAKE_DBG ?= ""
TESTS ?= tests/
FLAKE ?= flake8
NAME = arl
IMG ?= $(NAME)
TAG ?= ubuntu18.04
DOCKER_IMAGE = $(IMG):$(TAG)
BASE_IMAGE ?= piersharding/mpibase:ubuntu18.04
DOCKERFILE ?= Dockerfile
DOCKER = docker
DOCKER_REPO ?= piersharding/
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

# Kubernetes vars
KUBE_NAMESPACE ?= "default"
KUBECTL_VERSION ?= 1.14.1
HELM_VERSION ?= v2.14.0
HELM_CHART = arl-cluster
HELM_RELEASE ?= test

# ARL data directory usualy found in ./data
ARLDATA = $(CURRENT_DIR)/data

# INGRESS_HOST is the host name used in the Ingress resource definition for
# publishing services via the Ingress Controller
INGRESS_HOST ?= $(HELM_RELEASE).$(HELM_CHART).local
# define overides for above variables in here

-include PrivateRules.mak

.DEFAULT_GOAL := help

checkvars:
	@echo "Image: $(DOCKER_IMAGE)"
	@echo "Repo: $(DOCKER_REPO)"
	@echo "Host net device: $(SERVER_DEVICE)"
	@echo "Nfs: $(NFS_SERVER)"

all: clean build nosetests

docker_all: clean build docker_build docker_tests

clean: cleantests
	$(PYTHON) setup.py clean --all
	rm -f libarlffi.*.so
	rm -rf dist

in: inplace # just a shortcut
inplace:
	$(PYTHON) setup.py build >/dev/null 2>&1 && $(PYTHON) setup.py install >/dev/null 2>&1 || (echo "'$(PYTHON) setup.py install' failed."; exit -1)

build: in  ## build and install this project - make sure pipenv shell is activated

cleantests: ## clean out the cache before tests are run
	rm -rf coverage zernikes.png workers-*.dirlock
	cd tests && rm -rf __pycache__

unittest: cleantests  ## run tests using unittest
	MPLBACKEND=agg $(PYTHON) -m unittest -f --locals tests/*/test_*.py

pytest: cleantests  ## run tests using pytest
	pip install pytest >/dev/null 2>&1
	pytest -x $(TESTS)

nosetests: cleantests  ## run tests using nosetests
	rm -f predict_facet_timeslice_graph_wprojection.png pipelines-timings_*.csv
	ARL=$$(pwd) $(NOSETESTS) -s -v -e create_low_test_beam -e create_low_test_skycomponents_from_gleam $(TESTS)

nosetests-coverage: inplace cleantests  ## run nosetests with coverage
	rm -rf coverage .coverage
	ARL=$$(pwd) $(NOSETESTS) -s -v --with-coverage libs

trailing-spaces:
	find libs -name "*.py" -exec perl -pi -e 's/[ \t]*$$//' {} \;
	find processing_components -name "*.py" -exec perl -pi -e 's/[ \t]*$$//' {} \;
	find workflows -name "*.py" -exec perl -pi -e 's/[ \t]*$$//' {} \;

docs: inplace  ## build docs - you must have graphviz installed
	# you must have graphviz installed
	$(MAKE) -C docs dirhtml

code-flake:
	# flake8 ignore long lines and trailing whitespace
	$(FLAKE) --ignore=E501,W293,F401 --builtins=ModuleNotFoundError libs

code-lint:
	$(PYLINT) --extension-pkg-whitelist=numpy \
	  --ignored-classes=astropy.units,astropy.constants,HDUList \
	  -E libs/ tests/

code-analysis: code-flake code-lint  ## run pylint and flake8 checks

examples: inplace  ## launch examples
	$(MAKE) -C processing_library/notebooks
	$(MAKE) -C processing_components/notebooks
	$(MAKE) -C workflows/notebooks

notebook:  ## launch local jupyter notebook server
	DEVICE=`ip link | grep -E " ens| wlan| eth" | grep BROADCAST | tail -1 | cut -d : -f 2  | sed "s/ //"` && \
	IP=`ip a show $${DEVICE} | grep ' inet ' | awk '{print $$2}' | sed 's/\/.*//'` && \
	echo "Launching at IP: $${IP}" && \
	jupyter notebook --no-browser --ip=$${IP} --port=8888 examples/arl/

docker_build:
	$(DOCKER) build -t $(DOCKER_IMAGE) -f $(DOCKERFILE) \
	  --build-arg PYTHON=$(PYTHON) \
		--build-arg BASE_IMAGE=$(BASE_IMAGE) .

push: docker_build
	docker tag $(DOCKER_IMAGE) $(DOCKER_REPO)$(DOCKER_IMAGE)
	docker push $(DOCKER_REPO)$(DOCKER_IMAGE)

docker_nfs_arl_data:
	CONTAINER_EXISTS=$$($(DOCKER) ps -aqf ancestor=itsthenetwork/nfs-server-alpine) && \
	if [ -n "$${CONTAINER_EXISTS}" ]; then $(DOCKER) rm -f $${CONTAINER_EXISTS}; fi
	docker run -d --name nfs --privileged -p 2049:2049 \
	-v $(CURRENT_DIR)/:/arl \
	-e SHARED_DIRECTORY=/arl itsthenetwork/nfs-server-alpine:latest

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

namespace: ## create the kubernetes namespace
	kubectl describe namespace $(KUBE_NAMESPACE) || kubectl create namespace $(KUBE_NAMESPACE)

delete_namespace: ## delete the kubernetes namespace
	@if [ "default" == "$(KUBE_NAMESPACE)" ] || [ "kube-system" == "$(KUBE_NAMESPACE)" ]; then \
	echo "You cannot delete Namespace: $(KUBE_NAMESPACE)"; \
	exit 1; \
	else \
	kubectl describe namespace $(KUBE_NAMESPACE) && kubectl delete namespace $(KUBE_NAMESPACE); \
	fi

deploy: namespace mkcerts  ## deploy the helm chart
	@helm template charts/$(HELM_CHART)/ --name $(HELM_RELEASE) \
				 --namespace $(KUBE_NAMESPACE) \
         --tiller-namespace $(KUBE_NAMESPACE) \
				 --set helmTests=false \
				 --set arldatadir=$(ARLDATA) \
				 --set ingress.hostname=$(INGRESS_HOST) | kubectl -n $(KUBE_NAMESPACE) apply -f -

install: namespace mkcerts  ## install the helm chart (with Tiller)
	@helm tiller run $(KUBE_NAMESPACE) -- helm install charts/$(HELM_CHART)/ --name $(HELM_RELEASE) \
		--wait \
		--namespace $(KUBE_NAMESPACE) \
		--tiller-namespace $(KUBE_NAMESPACE) \
		--set arldatadir=$(ARLDATA) \
		--set ingress.hostname=$(INGRESS_HOST)

helm_delete: ## delete the helm chart release (with Tiller)
	@helm tiller run $(KUBE_NAMESPACE) -- helm delete $(HELM_RELEASE) --purge \
		--tiller-namespace $(KUBE_NAMESPACE)

show: mkcerts ## show the helm chart
	@helm template charts/$(HELM_CHART)/ --name $(HELM_RELEASE) \
				 --namespace $(KUBE_NAMESPACE) \
         --tiller-namespace $(KUBE_NAMESPACE) \
				 --set arldatadir=$(ARLDATA) \
				 --set ingress.hostname=$(INGRESS_HOST)

lint: ## lint check the helm chart
	@helm lint charts/$(HELM_CHART)/ \
				 --namespace $(KUBE_NAMESPACE) \
         --tiller-namespace $(KUBE_NAMESPACE) \
				 --set arldatadir=$(ARLDATA) \
				 --set ingress.hostname=$(INGRESS_HOST)

delete: ## delete the helm chart release
	@helm template charts/$(HELM_CHART)/ --name $(HELM_RELEASE) \
				 --namespace $(KUBE_NAMESPACE) \
         --tiller-namespace $(KUBE_NAMESPACE) | kubectl -n $(KUBE_NAMESPACE) delete -f -

logs: ## show Helm chart POD logs
	@for i in `kubectl -n $(KUBE_NAMESPACE) get pods -l app.kubernetes.io/instance=$(HELM_RELEASE) -o=name`; \
	do \
		echo "---------------------------------------------------"; \
		echo "Logs for $${i}"; \
		echo kubectl -n $(KUBE_NAMESPACE) logs $${i}; \
		echo kubectl -n $(KUBE_NAMESPACE) get $${i} -o jsonpath="{.spec.initContainers[*].name}"; \
		echo "---------------------------------------------------"; \
		for j in `kubectl -n $(KUBE_NAMESPACE) get $${i} -o jsonpath="{.spec.initContainers[*].name}"`; do \
			RES=`kubectl -n $(KUBE_NAMESPACE) logs $${i} -c $${j} 2>/dev/null`; \
			echo "initContainer: $${j}"; echo "$${RES}"; \
			echo "---------------------------------------------------";\
		done; \
		echo "Main Pod logs for $${i}"; \
		echo "---------------------------------------------------"; \
		for j in `kubectl -n $(KUBE_NAMESPACE) get $${i} -o jsonpath="{.spec.containers[*].name}"`; do \
			RES=`kubectl -n $(KUBE_NAMESPACE) logs $${i} -c $${j} 2>/dev/null`; \
			echo "Container: $${j}"; echo "$${RES}"; \
			echo "---------------------------------------------------";\
		done; \
		echo "---------------------------------------------------"; \
		echo ""; echo ""; echo ""; \
	done

describe: ## describe Pods executed from Helm chart
	@for i in `kubectl -n $(KUBE_NAMESPACE) get pods -l app.kubernetes.io/instance=$(HELM_RELEASE) -o=name`; \
	do echo "---------------------------------------------------"; \
	echo "Describe for $${i}"; \
	echo kubectl -n $(KUBE_NAMESPACE) describe $${i}; \
	echo "---------------------------------------------------"; \
	kubectl -n $(KUBE_NAMESPACE) describe $${i}; \
	echo "---------------------------------------------------"; \
	echo ""; echo ""; echo ""; \
	done

helm_tests:  ## run Helm chart tests
	helm tiller run $(KUBE_NAMESPACE) -- helm test $(HELM_RELEASE) --cleanup

helm_dependencies: ## Utility target to install Helm dependencies
	@which helm ; rc=$$?; \
	if [ $$rc != 0 ]; then \
	curl "https://kubernetes-helm.storage.googleapis.com/helm-$(HELM_VERSION)-linux-amd64.tar.gz" | tar zx; \
	mv linux-amd64/helm /usr/bin/; \
	helm init --client-only; \
	fi
	@helm init --client-only
	@if [ ! -d $$HOME/.helm/plugins/helm-tiller ]; then \
	echo "installing tiller plugin..."; \
	helm plugin install https://github.com/rimusz/helm-tiller; \
	fi
	helm version --client
	@helm tiller stop 2>/dev/null || true

kubectl_dependencies: ## Utility target to install K8s dependencies
	@([ -n "$(KUBE_CONFIG_BASE64)" ] && [ -n "$(KUBECONFIG)" ]) || (echo "unset variables [KUBE_CONFIG_BASE64/KUBECONFIG] - abort!"; exit 1)
	@which kubectl ; rc=$$?; \
	if [[ $$rc != 0 ]]; then \
		curl -L -o /usr/bin/kubectl "https://storage.googleapis.com/kubernetes-release/release/$(KUBERNETES_VERSION)/bin/linux/amd64/kubectl"; \
		chmod +x /usr/bin/kubectl; \
		mkdir -p /etc/deploy; \
		echo $(KUBE_CONFIG_BASE64) | base64 -d > $(KUBECONFIG); \
	fi
	@echo -e "\nkubectl client version:"
	@kubectl version --client
	@echo -e "\nkubectl config view:"
	@kubectl config view
	@echo -e "\nkubectl config get-contexts:"
	@kubectl config get-contexts
	@echo -e "\nkubectl version:"
	@kubectl version

localip:  ## set local Minikube IP in /etc/hosts file for Ingress $(INGRESS_HOST)
	@new_ip=`minikube ip` && \
	existing_ip=`grep $(INGRESS_HOST) /etc/hosts || true` && \
	echo "New IP is: $${new_ip}" && \
	echo "Existing IP: $${existing_ip}" && \
	if [ -z "$${existing_ip}" ]; then echo "$${new_ip} $(INGRESS_HOST)" | sudo tee -a /etc/hosts; \
	else sudo perl -i -ne "s/\d+\.\d+.\d+\.\d+/$${new_ip}/ if /$(INGRESS_HOST)/; print" /etc/hosts; fi && \
	echo "/etc/hosts is now: " `grep $(INGRESS_HOST) /etc/hosts`

mkcerts:  ## Make dummy certificates for $(INGRESS_HOST) and Ingress
	@if [ ! -f charts/$(HELM_CHART)/secrets/tls.key ]; then \
	openssl req -x509 -sha256 -nodes -days 365 -newkey rsa:2048 \
	   -keyout charts/$(HELM_CHART)/secrets/tls.key \
		 -out charts/$(HELM_CHART)/secrets/tls.crt \
		 -subj "/CN=$(INGRESS_HOST)/O=Minikube"; \
	else \
	echo "SSL cert already exits in charts/$(HELM_CHART)/secrets ... skipping"; \
	fi

help:  ## show this help.
	@echo "make targets:"
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ": .*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
	@echo ""; echo "make vars (+defaults):"
	@grep -E '^[0-9a-zA-Z_-]+ \?=.*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = " \\?\\= "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
