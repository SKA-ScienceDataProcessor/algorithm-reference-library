
Algorithm Reference Library on Kubernetes
=========================================

The following are a set of instructions of running the ARL on Kubernetes, and has been tested both on minikube v1.1.1, and a k8s cluster at version v1.14.3. on Ubuntu 18.04, and on Docker 18.09.6.

kubectl
=======

kubectl is the command line client for accessing the Kubernetes cluster.  This will need to be install locally, and be configured to point to your Kuberenetes cluster.  Installation instructions are here https://kubernetes.io/docs/tasks/tools/install-kubectl/.

Minikube
========

Using [Minikube](https://kubernetes.io/docs/getting-started-guides/minikube/) enables us to create a single node stand alone Kubernetes cluster for testing purposes.  If you already have a cluster at your disposal, then you can skip forward to 'Running the ARL on Kubernetes'.

The generic installation instructions are available at https://kubernetes.io/docs/tasks/tools/install-minikube/.

Minikube requires the Kubernetes runtime, and a host virtualisation layer such as kvm, virtualbox etc.  Please refer to the drivers list at https://github.com/kubernetes/minikube/blob/master/docs/drivers.md .

On Ubuntu 18.04, the most straight forward installation pattern is to go with the `none` driver as the host virtualisation layer as this means that it uses the host local dockerd and local file system is available for hostPath mounting.

The latest version of minikube is found here  https://github.com/kubernetes/minikube/releases .  Scroll down to the section for Linux, which will have instructions like:
```
curl -Lo minikube https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64 && chmod +x minikube && sudo mv minikube /usr/local/bin/
```

Now we need to bootstrap minikube and enable the ingress controller so that we have a running cluster based on `none`:
```
sudo minikube start \
    --extra-config=kubelet.resolv-conf=/run/systemd/resolve/resolv.conf \
    --vm-driver=none && sudo minikube addons enable ingress
```
This will take some time bootstrapping Kubernetes.  You will see output like the following when done.
```
Starting local Kubernetes v1.14.2 cluster...
Starting VM...
Getting VM IP address...
Moving files into cluster...
Setting up certs...
Connecting to cluster...
Setting up kubeconfig...
Starting cluster components...
Kubectl is now configured to use the cluster.
Loading cached images from config file.
...
✅  ingress was successfully enabled
```
Once completed, minikube will also update your kubectl settings to include the context `current-context: minikube` in `~/.kube/config`.  Test that connectivity works with something like:
```
kubectl get pods
# outputs:
#NAME            READY     STATUS    RESTARTS   AGE
#busybox-fmlp5   1/1       Running   0          10m
```
Next, build the ARL image:
```
make docker_build
```
The build output will follow:
```
docker build -t arl:ubuntu18.04 -f Dockerfile \
  --build-arg PYTHON=python3 \
	--build-arg BASE_IMAGE=piersharding/mpibase:ubuntu18.04 .
Sending build context to Docker daemon  18.56MB
Step 1/38 : ARG BASE_IMAGE=ubuntu:18.04
Step 2/38 : FROM $BASE_IMAGE
 ---> ab943c833399
Step 3/38 : ARG PYTHON=python3
...
```

Test that this is correctly accessible by launching a test app:
```
kubectl run test-arl --rm -it --image arl:ubuntu18.04 --image-pull-policy=IfNotPresent -- bash
```
Exit the shell, and then clean up the app (if necessary):
```
kubectl delete deployment test-arl
```

Once the ARL containers are running as described below, we will need to know the vm IP address so that we can reach services that are running inside the VM via the Ingress.

```
$ make localip
New IP is: 192.168.86.47
Existing IP: 192.168.86.47 test.arl-cluster.local
/etc/hosts is now:  192.168.86.47 test.arl-cluster.local
```

Cleaning up:
```
sudo minikube stop # stop minikube - this can be restarted with minikube start
sudo minikube delete # destroy minikube - totally gone!
```

Running the ARL on Kubernetes
-------------------------------
The ARL has been containerised, and requires three versions of the container to run:

* Jupyter notebook
* Dask Scheduler
* 1 or more Dask workers

The ARL containers also need access to the test data which is mounted into the appropriate containers at /arl/data .

```
git lfs pull # ensure the visibility data is checked out
```

Helm Chart
----------

First you must install [Helm](https://docs.helm.sh/using_helm/#installing-helm), the easiest way is using make:
```
make helm_dependencies
```
Once Helm is up and running, change to the charts/arl-cluster/ and check the values in in the values.yaml file, in particular the following:
```
...
worker:
  replicaCount: 1

image:
  repository: arl
  tag: ubuntu18.04
  pullPolicy: IfNotPresent

jupyter:
  password: changeme

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
make install
```
You will get output similar to the following:
```
$ make install
kubectl describe namespace "default" || kubectl create namespace "default"
Name:         default
Labels:       <none>
Annotations:  <none>
Status:       Active

No resource quota.

No resource limits.
SSL cert already exits in charts/arl-cluster/secrets ... skipping
Installed Helm version v2.13.1
Installed Tiller version v2.13.1
Helm and Tiller are the same version!
Creating tiller namespace (if missing): default
Starting Tiller...
Tiller namespace: default
Running: helm install charts/arl-cluster/ --name test --wait --namespace default --set arldatadir=/home/piers/git/private/algorithm-reference-library/data --set ingress.hostname=test.arl-cluster.local

NAME:   test
LAST DEPLOYED: Tue Jun 11 09:41:37 2019
NAMESPACE: default
STATUS: DEPLOYED

RESOURCES:
==> v1/Deployment
NAME                             READY  UP-TO-DATE  AVAILABLE  AGE
dask-scheduler-arl-cluster-test  1/1    1           1          80s
dask-worker-arl-cluster-test     3/3    3           3          80s
notebook-arl-cluster-test        1/1    1           1          80s

==> v1/PersistentVolume
NAME                      CAPACITY  ACCESS MODES  RECLAIM POLICY  STATUS  CLAIM                             STORAGECLASS  REASON  AGE
arldata-arl-cluster-test  1Gi       RWX           Retain          Bound   default/arldata-arl-cluster-test  standard      80s

==> v1/PersistentVolumeClaim
NAME                      STATUS  VOLUME                    CAPACITY  ACCESS MODES  STORAGECLASS  AGE
arldata-arl-cluster-test  Bound   arldata-arl-cluster-test  1Gi       RWX           standard      80s

==> v1/Pod(related)
NAME                                              READY  STATUS   RESTARTS  AGE
dask-scheduler-arl-cluster-test-656b5df8f7-9mkvk  1/1    Running  0         80s
dask-worker-arl-cluster-test-74597ffc56-57jc4     1/1    Running  0         80s
dask-worker-arl-cluster-test-74597ffc56-5l7xg     1/1    Running  0         80s
dask-worker-arl-cluster-test-74597ffc56-dh28c     1/1    Running  0         80s
notebook-arl-cluster-test-bf7597777-nrw7m         1/1    Running  0         80s

==> v1/Secret
NAME                 TYPE               DATA  AGE
tls-secret-arl-test  kubernetes.io/tls  2     80s

==> v1/Service
NAME                             TYPE       CLUSTER-IP    EXTERNAL-IP  PORT(S)            AGE
dask-scheduler-arl-cluster-test  ClusterIP  10.98.57.191  <none>       8786/TCP,8787/TCP  80s
notebook-arl-cluster-test        ClusterIP  10.102.42.14  <none>       8888/TCP           80s

==> v1beta1/Ingress
NAME                       HOSTS                   ADDRESS        PORTS    AGE
notebook-arl-cluster-test  test.arl-cluster.local  192.168.86.47  80, 443  80s


Stopping Tiller...
```
Point your browser at https://test.arl-cluster.local for accessing the Jupyter Notebook service
