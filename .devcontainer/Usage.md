In the root directory of repository:

## Build Docker image

```bash
# Change UID and GID so that volume sharing works
docker build --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g) -f .devcontainer/Dockerfile -t simtech/symplectic_nn:latest .
```

## Run experiments

```bash
HOST_DIR="/usr/local/faststorage/muellese/dev/simtech-bachelorarbeit" # change to your location
CONTAINER_DIR="/workspaces/simtech-bachelorarbeit"
docker run --rm -v $HOST_DqIR:$CONTAINER_DIR -w $CONTAINER_DIR simtech/symplectic_nn:latest /bin/bash -c "pip install -e . && cd demos && OMP_NUM_THREADS=4 python lowdim.py --epochs 200"
```

or to run scripts
```bash
docker run --rm -v $HOST_DIR:$CONTAINER_DIR -w $CONTAINER_DIR simtech/symplectic_nn:latest /bin/bash -c "pip install -e . && cd scripts && OMP_NUM_THREADS=4 python thesis_lowdim.py run-experiments 2>&1 | tee 2020-11-15-lowdim.log"
```

# Run Tensorboard

Tensorboard can be accessed on the host on `localhost:6006`

```bash
HOST_DIR="/usr/local/faststorage/muellese/dev/simtech-bachelorarbeit" # change to your location
CONTAINER_DIR="/workspaces/simtech-bachelorarbeit"
docker run --rm -p 127.0.0.1:6007:6007 -v $HOST_DIR:$CONTAINER_DIR -w $CONTAINER_DIR simtech/symplectic_nn:latest /bin/bash -c "pip install -e . && cd demos && tensorboard --bind_all --logdir runs --port 6007"
```