# SimTech Bachelorthesis 
## Symplectic Neural Networks

The final thesis can be found under `latex/thesis/thesis_FINAL.pdf`.
This repository contains the implementation of the symplectic networks (SympNets)
described in the thesis.

## Installation

The easiest way to install all requirements is to use Visual Studio Code's
development container functionality (https://code.visualstudio.com/docs/remote/containers).
When opening the root folder in Visual Studio Code, there will be
a prompt to open the project in a development container, because the folder
contains a dev container configuration `.devcontainer/devcontainer.json`.
Visual Studio Code will automatically setup the development
environment based on this configuration. This process requires a working Docker 
installation on the local computer.

Alternatively, the Docker container can be invoked standalone, 
see `.devcontainer/Usage.md` for sample instructions.

As another alternative, the dependencies can be installed manually.
The file `requirements.txt` contains all required Python dependencies and can
be consumed by Python's `pip` package manager. In addition,
the local root folder has to be installed as a package via `$ pip install -e . `.
This suffices for everything except for the three-dimensional models 
(which were not used in the thesis at the end). 
If one wants to run the three-dimensional models, more dependencies (Dolfin/Fenics) 
have to be installed. We refer to `.devcontainer/Dockerfile` for details.
We recommend to setup a isolated Python environment with 
https://docs.python.org/3/tutorial/venv.html for a manual installation, or similar.

## Structure and usage
The individual symplectic layer implementations can be found under `/nn`:
- `nn/linearsymplectic.py` contains all linear symplectic layers.
- `nn/nonlniearsymplectic.py` contains all non-linear symplectic layers.
- `nn/models.py` contains some wrapper classes and helper methods.
- `nn/symplecticloss.py` contains a implementation of a symplectic L2 soft loss 
(only used for debugging purposes, and not as a training loss).

The folder `/tests` contains unit tests for some layers. The unit tests can be run
in the folder with for example `$ python test_conv1d.py`.

The folder `/models` refers to a git submodule which contains
the base model implementations (for example Harmonic Oscillator,
Simple Pendulum, Sine-Gordon, ...).

The folder `/experiments` contains scripts for the low-dimensional
(`lowdim.py`) and high-dimensional (`wave.py`) experiments. In the folder, run
`$ python lowdim.py --help` or `$ python wave.py --help` to get an overview
over all configurable experiment parameters.

In the thesis, we have run several low-dimensional and high-dimensional experiments
with different sets of parameters. Under `/scripts`, we have implemented
orchestration scripts, which automatically run all experiments conducted in
the thesis by orchestrating the scripts mentioned in the previous paragraph. 
The process is split into two commands:

For the low-dimensional experiments:
- `$ thesis-lowdim.py run-experiments` runs the low-dimensional scripts.
The resulting data is stored in `/scripts/data-lowdim`.
- `$ thesis-lowdim.py update-csv` extracts the relevant data to csv files
in `latex/data`, where it is used from Latex via pgfplots 
(https://ctan.org/pkg/pgfplots?lang=en).

For the high-dimensional experiments:
- `$ thesis-wave.py run-experiments` runs the low-dimensional scripts.
The resulting data is stored in `/scripts/data-wave`.
- `$ thesis-wave.py update-csv` extracts the relevant data to csv files
in `latex/data-wave`, where it is used from Latex via pgfplots 
(https://ctan.org/pkg/pgfplots?lang=en).

Both scripts allow parallel execution of multiple experiments.
The parallelism is controled with the `--parallel` argument, for example
`--parallel 4` would run four experiments at the same time.

The output of `run-experiments` contains Tensorboard 
(https://pytorch.org/docs/stable/tensorboard.html) log data for every
experiment in folders called `tensorboard`.

The folder `/latex` contains the Latex code for the thesis.
The folder `/meetings` serves as a archive of code implemented
for the first meetings in the project.