#!/bin/sh

# launch the notebook - the IP address to listen on is passed in via env-var IP
IP=${IP:-0.0.0.0}
jupyter notebook --allow-root --no-browser --ip=${IP} --port=8888 /arl/examples/arl/
