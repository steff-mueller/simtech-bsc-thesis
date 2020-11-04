#!/bin/bash
python lowdim.py \
    --model simple_pendulum \
    --qmin 3 \
    --qmax 20 \
    --pmin 0.5 \
    --pmax 2.5 \
    -n 400 \
    "$@"