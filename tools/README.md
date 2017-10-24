
Docker Container Deployment
===========================

The tools directory contains a simple [Ansible playbook](https://www.ansible.com/) for deploying a Dask cluster.
The playbook, docker.yml, has been tested with Ansible 2.4.0.  It installs the docker-py dependency on the remote hosts which is used to manipulate Docker containers.
In order for this to work, the tools/inventory/docker, and tools/ssh.config files must be amended to point to your hosts.
Ensure that you specify one [master] node.
Update the tools/inventory/group_vars/all file to point to your Docker repository for the arl_image and arl_image_tag variables, based on the running of:
```
make docker_push PYTHON=python3 DOCKER_REPO=localhost:5000 # change to point to your repository
```

The playbook is launched using:
```
make launch_dask
```
