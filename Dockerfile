# Universal image for running Notebook, Dask pipelines, tests, and lint checkers
ARG PYTHON=python3.6

FROM ubuntu:17.04

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
            python-software-properties build-essential curl wget fonts-liberation ca-certificates && \
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
    apt-get clean -y

# sort out pip and python for 3.6
RUN cd /src; wget https://bootstrap.pypa.io/get-pip.py && ${PYTHON} get-pip.py; \
    rm -rf /root/.cache

RUN if [ "${PYTHON}" = "python3.6" ] ; then ln -s /usr/bin/python3.6 /usr/local/bin/python3 && \
    rm -f /usr/bin/python3 && ln -s /usr/bin/python3.6 /usr/bin/python3 ; fi

# Install Tini
RUN wget --quiet https://github.com/krallin/tini/releases/download/v0.10.0/tini && \
    echo "1361527f39190a7338a0b434bd8c88ff7233ce7b9a4876f3315c22fce7eca1b0 *tini" | sha256sum -c - && \
    mv tini /usr/local/bin/tini && \
    chmod +x /usr/local/bin/tini

# Add and install Python modules
ADD ./requirements.txt /src/requirements.txt
RUN cd /src; pip3 install -r requirements.txt; rm -rf /root/.cache
RUN pip3 install bokeh && pip3 install pytest; pip3 install jupyter_nbextensions_configurator; pip3 install jupyter_contrib_nbextensions; rm -rf /root/.cache
RUN pip3 install -U pylint; rm -rf /root/.cache
RUN jupyter contrib nbextension install --system --symlink
RUN jupyter nbextensions_configurator enable --system

# runtime specific environment
ENV JENKINS_URL 1
ENV PYTHONPATH /arl
ENV JUPYTER_PATH /arl/examples/arl

# Bundle app source
ADD ./docker/boot.sh /
ADD ./arl /arl/arl/
ADD ./Makefile /arl/
ADD ./examples /arl/examples/
ADD ./tests /arl/tests/

# create space for tests
RUN mkdir -p /arl/test_data /arl/test_results && \
    chmod 777 /arl /arl/test_data /arl/test_results && \
    chmod -R a+w /arl

# We share in the arl data here
VOLUME ["/arl/data"]

# Expose Jupyter and Bokeh ports
EXPOSE  8888 8787

# Setup the entrypoint or environment
ENTRYPOINT ["tini", "--"]

# Run - default is notebook
CMD ["/boot.sh"]
