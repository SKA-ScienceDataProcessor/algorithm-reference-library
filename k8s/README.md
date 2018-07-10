
Algorithm Reference Library on Kubernetes
=========================================

The following are a set of instructions of running the ARL on Kubernetes, and has been tested both on minikube, and a k8s cluster at version v1.10.2. on Ubuntu 18.04.

kubectl
=======

kubectl is the command line client for accessing the Kubernetes cluster.  This will need to be install locally, and be configured to point to your Kuberenetes cluster.  Installation instructions are here https://kubernetes.io/docs/tasks/tools/install-kubectl/.

Minikube
========

Using [Minikube](https://kubernetes.io/docs/getting-started-guides/minikube/) enables us to create a single node stand alone Kubernetes cluster for testing purposes.  If you already have a cluster at your disposal, then you can skip forward to 'Running the ARL on Kubernetes'.

The generic installation instructions are available at https://kubernetes.io/docs/tasks/tools/install-minikube/.

Minikube requires the Kubernetes runtime, and a host virtualisation layer such as kvm, virtualbox etc.  Please refer to the drivers list at https://github.com/kubernetes/minikube/blob/master/docs/drivers.md .

On Ubuntu 18.04, the most straight forward installation pattern is to go with kvm as the host virtualisation layer, and use the kvm2 driver.
To install [kvm](http://www.linux-kvm.org/page/Main_Page) on Ubuntu it should be simply a case of:
```
sudo apt-get install qemu-kvm libvirt-bin ubuntu-vm-builder bridge-utils
```
The detailed instructions can be found here https://help.ubuntu.com/community/KVM/Installation.

Once kvm is installed, we need to install the kvm2 driver with:
```
curl -LO https://storage.googleapis.com/minikube/releases/latest/docker-machine-driver-kvm2 && chmod +x docker-machine-driver-kvm2 && sudo mv docker-machine-driver-kvm2 /usr/local/bin/
```

Once the kvm2 driver is installed, then the latest version of minikube is found here  https://github.com/kubernetes/minikube/releases .  Scroll down to the section for Linux, which will have instructions like:
```
curl -Lo minikube https://storage.googleapis.com/minikube/releases/v0.27.0/minikube-linux-amd64 && chmod +x minikube && sudo mv minikube /usr/local/bin/
```

Now we need to bootstrap minikube so that we have a running cluster based on kvm:
```
minikube start --vm-driver kvm2
```
This will take some time setting up the vm, and bootstrapping Kubernetes.  You will see output like the following when done.
```
Starting local Kubernetes v1.10.0 cluster...
Starting VM...
Getting VM IP address...
Moving files into cluster...
Setting up certs...
Connecting to cluster...
Setting up kubeconfig...
Starting cluster components...
Kubectl is now configured to use the cluster.
Loading cached images from config file.
```
Once completed, minikube will also update your kubectl settings to include the context `current-context: minikube` in `~/.kube/config`.  Test that connectivity works with something like:
```
kubectl get pods
# outputs:
#NAME            READY     STATUS    RESTARTS   AGE
#busybox-fmlp5   1/1       Running   0          10m
```
Next, we need to ensure that the ARL Docker image is available to the minikube cluster.  The simplest way to do this is to use the minikube/Docker feature for connection to the Docker daemon inside the VM, and then rebuild the ARL image:
```
eval $(minikube docker-env)
env | grep DOCKER
#DOCKER_CERT_PATH=/home/piers/.minikube/certs
#DOCKER_TLS_VERIFY=1
#DOCKER_HOST=tcp://192.168.39.37:2376
#DOCKER_API_VERSION=1.23

make docker_build
```
The build output will follow:
```
docker build -t arl_img:latest -f Dockerfile --build-arg PYTHON=python3.6 .
Sending build context to Docker daemon  4.771MB
Step 1/44 : ARG PYTHON=python3.6
Step 2/44 : FROM ubuntu:18.04
18.04: Pulling from library/ubuntu
...
```

Alternatively we can use registry credentials declared in the resource descriptor to pull the image from remote:
```
...
      imagePullSecrets:
        - name: arlregcred
...
```
To enable this we need to create a Kubernetes Secret that holds the container registry credentials called arlregcred:
```
kubectl create secret docker-registry arlregcred --docker-server=<registry.domain.name> --docker-username=<username> --docker-password=<password> --docker-email=<you@email.address?
```

Test that this is correctly accessible by launching a test app:
```
kubectl run test-arl --rm -it --image arl_img:latest --image-pull-policy=IfNotPresent -- bash
```
Exit the shell, and then clean up the app (if necessary):
```
kubectl delete deployment test-arl
```

Once the ARL containers are running as described below, we will need to know the vm IP address so that we can reach services that are running inside the VM on the Pod network.

```
minikube ip
# where minikube ip gives the IP address of the minikube vm
```

Cleaning up:
```
minikube stop # stop minikube - this can be restarted with minikube start
minikube delete # destroy minikube - totally gone!
```

Running the ARL on Kubernetes
-------------------------------
The ARL has been containerised, and requires three versions of the container to run:

* Jupyter notebook
* Dask Scheduler
* 1 or more Dask workers

The ARL containers also need access to the test data which is mounted into the appropriate containers at /arl/data .  The easiest way to share this data in is using NFS.  This is conveniently done with the docker+nfs container which is bootstrapped using the Makefile.  The following assumes that you are running docker on a network accessible host to the Kubernetes cluster (localhost in the case of minikube - ensure that the minikube docker-env has NOT been set in the current shell as you will end up pointing at the wrong dockerd).

```
git lfs pull # ensure the visibility data is checked out
make docker_nfs_arl_data
```
Output will be something like as follows:
```
docker run -d --name nfs --privileged -p 2049:2049 \
-v /home/piers/git/private/algorithm-reference-library/:/arl \
-e SHARED_DIRECTORY=/arl itsthenetwork/nfs-server-alpine:latest
79351289297f54cbcc2e960c0e09143d3f661c96342f0b4e00d18d72d148281c
```
Choices
-------

There are two choices for running the ARL on Kubernetes :-
* Resource descriptors
* Helm Chart


Resource Descriptors
--------------------
For each of the above describe container types, there is a separate resource descriptor file for - scheduler, worker, and notebook.  These can be found in the [k8s/resources/](k8s/resources/) directory.
Launch them all with:
```
make k8s_deploy DOCKER_REPO="" DOCKER_IMAGE=arl_img:latest
```
Setting DOCKER_REPO="" will assume that the image in DOCKER_IMAGE is available on each node of the cluster, which is correct for the minikube example.  For yur own cluster, this will need to be adjusted accordingly, as well as ensuring that `docker login` has been performed on each node as required (if the repository is private) and remember dockerd parameters such as --insecure-registry may also be required (https://docs.docker.com/registry/insecure/).

There are a number of other variables that can be passed to `make` - see the Makefile for more details, but the main defaults are:
```
WORKER_MEM ?= 512Mi
WORKER_CPU ?= 500m
WORKER_REPLICAS ?= 1 # only one instance of the Dask Worker
```

Check the apps are running with:
```
kubectl get pods
#NAME                              READY     STATUS    RESTARTS   AGE
#busybox-fmlp5                     1/1       Running   0          56m
#dask-scheduler-8487d4bc97-5vxjb   0/1       Running   0          4s
#dask-worker-86b85f9558-r7dff      0/1       Running   0          4s
#notebook-857c6fdcdc-4cmx6         0/1       Running   0          4s
```

In order to connect to the notebook, and Dask scheduler, we need to find the service node ports that are being used:
```
 kubectl get services
NAME             TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)                         AGE
dask-scheduler   NodePort    10.107.52.192   <none>        8786:31375/TCP,8787:30861/TCP   3m
kubernetes       ClusterIP   10.96.0.1       <none>        443/TCP                         1h
notebook         NodePort    10.109.136.48   <none>        8888:32123/TCP                  3m
```
In this case, the Jupyter notebook server is available at http://<kubernetes master IP>:32123, and the Dask scheduler bokeh server is on http://<kubernetes master IP>:30861 (kubernetes master IP can be found with `minikube ip` for minikube).

Login into the Jupyter notebook server - the default password is 'changeme'.  avigate to workflows/notebooks and find imaging-pipelines.ipynb to test the orchestrated environment.

Tear down the test with:
```
make k8s_delete DOCKER_REPO="" DOCKER_IMAGE=arl_img:latest
```

Helm Chart
----------

First you must install [Helm](https://docs.helm.sh/using_helm/#installing-helm), the easiest way is using the install script:
```
https://raw.githubusercontent.com/kubernetes/helm/master/scripts/get | bash
```
You must initialise Helm, with `helm init`.  This will ensure that the [Tiller](https://docs.helm.sh/glossary/#tiller) component is running which is the Kubernetes API server proxy for Helm.  Check this is running correctly with `helm version`.

Once Helm is up and running, change to the k8s/arl-cluster/ and check the values in in the values.yaml file, in particular the following:
```
...
worker:
  replicaCount: 1

image:
  repository: arl_img
  tag: latest
  pullPolicy: IfNotPresent

jupyter:
  password: changeme

nfs:
  server: 192.168.0.168
...
resources:
  limits:
   cpu: 500m     # 500m = 0.5 CPU
   memory: 512Mi # 512Mi = 0.5 GB mem
...
```
As with the above instructions for Resource Descriptors, the image location must be set to something accessible by every node in Kubernetes.
Change the worker.replicaCount and resources values to something more desirable for your cluster.

Change directory back to k8s/ and launch helm:
```
helm install --name <app instance> arl-cluster/
```
Individual values from the values.yaml file can be overridden with: `--set worker.replicaCount=10,resource.limits.cpu=1000m,resource.limits.memory=4098Mi` etc.

You will get output like the following (adjust for your registry and credentials):
```
k8s$ helm install --name test arl-cluster/ --wait \
      --set image.repository="registry.gitlab.com/piersharding/arl-devops/arl_img" \
      --set image.tag="1b47647d" \
      --set image.secrets[0].name="gitlab-registry"
NAME:   test
LAST DEPLOYED: Fri Jun  1 10:02:58 2018
NAMESPACE: default
STATUS: DEPLOYED

RESOURCES:
==> v1/Service
NAME                             TYPE       CLUSTER-IP     EXTERNAL-IP  PORT(S)            AGE
notebook-test-arl-cluster        ClusterIP  10.108.56.174  <none>       8888/TCP           0s
dask-scheduler-test-arl-cluster  ClusterIP  10.101.121.34  <none>       8786/TCP,8787/TCP  0s

==> v1/Deployment
NAME                          DESIRED  CURRENT  UP-TO-DATE  AVAILABLE  AGE
notebook                      1        1        1           0          0s
dask-scheduler                1        1        1           0          0s
dask-worker-test-arl-cluster  1        1        1           0          0s

==> v1/Pod(related)
NAME                                           READY  STATUS             RESTARTS  AGE
notebook-565795c79f-rgjxg                      0/1    ContainerCreating  0         0s
dask-scheduler-68dbfd8fbb-hl6g6                0/1    ContainerCreating  0         0s
dask-worker-test-arl-cluster-6f848cdb6d-2x5v2  0/1    ContainerCreating  0         0s


NOTES:
Get the Jupyter Notebook application URL by running these commands:
1. Calculate and export the POD_NAME:
  export POD_NAME=$(kubectl get pods --namespace default -l "app=notebook-arl-cluster,release=test" -o jsonpath="{.items[0].metadata.name}")
2. Forward local port 8080 to Jupyter on the POD with:
  kubectl port-forward $POD_NAME 8080:8888
3. Visit http://127.0.0.1:8080 to use your application
```
Follow the NOTES instructions for accessing the Jupyter Notebook service
