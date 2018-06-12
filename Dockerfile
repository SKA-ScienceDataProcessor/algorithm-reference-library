# Universal image for running Notebook, Dask pipelines, libs, and lint checkers
ARG PYTHON=python3.6

FROM ubuntu:18.04

ARG PYTHON

MAINTAINER Piers Harding "piers@catalyst.net.nz"

ENV LANG en_NZ.UTF-8
ENV LANGUAGE en_NZ.UTF-8
ENV LC_ALL en_NZ.UTF-8
ENV HOME /root
ENV DEBIAN_FRONTEND noninteractive

# the package basics for Python 3
RUN \
    DEBIAN_FRONTEND=noninteractive apt-get update -y && \
    apt-get install -y locales tzdata python-pip net-tools vim-tiny software-properties-common \
            python3-software-properties build-essential curl wget fonts-liberation ca-certificates && \
    echo "Setting locales  ..." && /usr/sbin/locale-gen en_US.UTF-8 && \
    /usr/sbin/locale-gen en_NZ.UTF-8 && \
    echo "Setting timezone ..." &&  /bin/echo 'Pacific/Auckland' | tee /etc/timezone && DEBIAN_FRONTEND=noninteractive dpkg-reconfigure --frontend noninteractive tzdata && \
    add-apt-repository -y ppa:git-core/ppa && \
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install -y git-lfs && \
    git lfs install && \
    apt-get install -y ${PYTHON}-dev python3-tk flake8 python3-nose \
            virtualenv virtualenvwrapper && \
    apt-get install -y graphviz && \
    apt-get install -y nodejs npm && \
    apt-get clean -y

# node node is linked to nodejs
RUN if [ ! -f /usr/bin/node ]; then ln -s /usr/bin/nodejs /usr/bin/node ; fi && \
    node --version

# sort out pip and python for 3.6
RUN cd /src; wget https://bootstrap.pypa.io/get-pip.py && ${PYTHON} get-pip.py; \
    rm -rf /root/.cache

RUN if [ "${PYTHON}" = "python3.6" ] ; then ln -s /usr/bin/python3.6 /usr/local/bin/python3 && \
    rm -f /usr/bin/python3 && ln -s /usr/bin/python3.6 /usr/bin/python3 ; fi && \
    python3 --version && \
    pip3 --version


# Install Tini
RUN wget --quiet https://github.com/krallin/tini/releases/download/v0.18.0/tini && \
    echo "12d20136605531b09a2c2dac02ccee85e1b874eb322ef6baf7561cd93f93c855 *tini" | sha256sum -c - && \
    mv tini /usr/local/bin/tini && \
    chmod +x /usr/local/bin/tini

# Add and install Python modules
ADD ./requirements.txt /src/requirements.txt
RUN cd /src; pip3 install -r requirements.txt; rm -rf /root/.cache
RUN pip3 install bokeh && pip3 install pytest; pip3 install jupyter_nbextensions_configurator; pip3 install jupyter_contrib_nbextensions; rm -rf /root/.cache
RUN pip3 install -U pylint; rm -rf /root/.cache
RUN jupyter contrib nbextension install --system --symlink
RUN jupyter nbextensions_configurator enable --system
#RUN pip install jupyterlab
#RUN jupyter serverextension enable --py jupyterlab --sys-prefix
#RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager
#RUN jupyter labextension install jupyterlab_bokeh

# runtime specific environment
ENV JENKINS_URL 1
ENV PYTHONPATH /arl
ENV JUPYTER_PATH /arl/examples/arl

RUN touch "${HOME}/.bash_profile"

# Bundle app source
ADD ./docker/boot.sh /
ADD ./libs /arl/libs/
ADD ./Makefile /arl/
ADD ./examples /arl/examples/
ADD ./tests /arl/tests/
ADD ./setup.py ./README.md /arl/
ADD ./data_models /arl/data_models/
ADD ./processing_components /arl/processing_components/
ADD ./util /arl/util/
ADD ./workflows /arl/workflows/

# run setup
RUN cd /arl && ${PYTHON} setup.py build && ${PYTHON} setup.py install

# create space for libs
RUN mkdir -p /arl/test_data /arl/test_results && \
    chmod 777 /arl /arl/test_data /arl/test_results && \
    chmod -R a+w /arl

COPY --chown="1000:100" ./docker/jupyter_notebook_config.py "${HOME}/.jupyter/"
COPY ./docker/notebook.sh /usr/local/bin/
COPY ./docker/start-dask-scheduler.sh /usr/local/bin/
COPY ./docker/start-dask-worker.sh /usr/local/bin

# We share in the arl data here
VOLUME ["/arl/data", "/arl/tmp"]

# Expose Jupyter and Bokeh ports
EXPOSE  8888 8786 8787 8788 8789

# Setup the entrypoint or environment
ENTRYPOINT ["tini", "--"]

# Run - default is notebook
CMD ["/boot.sh"]
