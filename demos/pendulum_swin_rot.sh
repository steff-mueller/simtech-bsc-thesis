#!/bin/bash
python lowdim.py \
    --model simple_pendulum \
    --qmin -20 \
    --qmax 20 \
    --pmin -2.5 \
    --pmax 2.5 \
    -n 800 \
    "$@"