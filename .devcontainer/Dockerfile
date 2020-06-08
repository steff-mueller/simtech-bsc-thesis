FROM ubuntu:20.04

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# Or your actual UID, GID on Linux if not the default 1000
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

ENV PYBIND11_VERSION=2.2.3

RUN apt update  \
    && apt-get install -y --no-install-recommends python3 python3-virtualenv python3-dev git wget build-essential cmake libboost-all-dev libeigen3-dev petsc-dev \
    #
    # Create a non-root user to use if preferred - see https://aka.ms/vscode-remote/containers/non-root-user.
    && groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME
    #

# from https://pythonspeed.com/articles/activate-virtualenv-dockerfile/
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m virtualenv --python=/usr/bin/python3 $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# # https://bitbucket.org/fenics-project/dolfin/issues/1069/petsc-311-patch-vecscattercreatewithdata
# sympy==1.2 does not work!
RUN PETSC_DIR=/usr/lib/petscdir/3.12 pip install petsc4py==3.12.0 fenics-ffc==2019.1.0.post0 sympy==1.1.1

WORKDIR /tmp/fenics-build
RUN wget -nc --quiet https://github.com/pybind/pybind11/archive/v${PYBIND11_VERSION}.tar.gz \
    && tar -xf v${PYBIND11_VERSION}.tar.gz

WORKDIR /tmp/fenics-build/pybind11-${PYBIND11_VERSION}/build
RUN cmake -DCMAKE_INSTALL_PREFIX=$VIRTUAL_ENV -DPYBIND11_TEST=off .. \
    && make install

WORKDIR /tmp/fenics-build
RUN FENICS_VERSION=$(python -c"import ffc; print(ffc.__version__)") \
    && git clone --branch=$FENICS_VERSION https://bitbucket.org/fenics-project/dolfin

WORKDIR /tmp/fenics-build/dolfin/build
RUN cmake ../ -DCMAKE_INSTALL_PREFIX=$VIRTUAL_ENV \
    && make \
    && make install

WORKDIR /tmp//fenics-build
RUN pip install dolfin/python

# Install other python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt