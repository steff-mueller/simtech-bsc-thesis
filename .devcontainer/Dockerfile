FROM ubuntu:20.04

# Avoid warnings by switching to noninteractive.
ENV DEBIAN_FRONTEND=noninteractive

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
ARG LATEX

ENV PYBIND11_VERSION=2.2.3

RUN apt update  \
    && apt-get install -y --no-install-recommends python3 python3-virtualenv python3-dev git wget build-essential cmake libboost-all-dev libeigen3-dev petsc-dev \
    && if [ "$LATEX" = "yes" ] ; then apt-get install -y texlive-latex-extra latexmk ; fi \
    #
    # Create a non-root user to use if preferred - see https://aka.ms/vscode-remote/containers/non-root-user.
    && groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # Add sudo support.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Change user
USER ${USERNAME} 

# Create and activate Python virtual environment.
# See https://pythonspeed.com/articles/activate-virtualenv-dockerfile/
ENV VIRTUAL_ENV=/home/vscode/venv
RUN python3 -m virtualenv --python=/usr/bin/python3 $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV PKG_CONFIG_PATH="$VIRTUAL_ENV/lib/pkgconfig/"

# https://bitbucket.org/fenics-project/dolfin/issues/1069/petsc-311-patch-vecscattercreatewithdata
ENV FENICS_BUILD="/tmp/fenics-build"
RUN PETSC_DIR=/usr/lib/petscdir/3.12 pip install petsc4py==3.12.0 fenics-ffc==2019.1.0.post0 \
    #
    # Install pybind11.
    && mkdir ${FENICS_BUILD} \
    && cd ${FENICS_BUILD} \
    && wget -nc --quiet https://github.com/pybind/pybind11/archive/v${PYBIND11_VERSION}.tar.gz \
    && tar -xf v${PYBIND11_VERSION}.tar.gz \
    #
    && mkdir ${FENICS_BUILD}/pybind11-${PYBIND11_VERSION}/build \
    && cd ${FENICS_BUILD}/pybind11-${PYBIND11_VERSION}/build \
    && cmake -DCMAKE_INSTALL_PREFIX=$VIRTUAL_ENV -DPYBIND11_TEST=off .. \
    && make install \
    #
    # Install dolfin.
    && cd ${FENICS_BUILD} \
    && FENICS_VERSION=$(python -c"import ffc; print(ffc.__version__)") \
    && git clone --branch=$FENICS_VERSION https://bitbucket.org/fenics-project/dolfin \
    #
    && mkdir ${FENICS_BUILD}/dolfin/build \
    && cd ${FENICS_BUILD}/dolfin/build \
    && cmake ../ -DCMAKE_INSTALL_PREFIX=$VIRTUAL_ENV \
    && make \
    && make install \
    #
    && cd ${FENICS_BUILD} \
    && pip install dolfin/python \
    #
    # Clean up
    && rm -rf ${FENICS_BUILD}

# Install Python dependencies.
COPY requirements.txt .
RUN pip install -r requirements.txt