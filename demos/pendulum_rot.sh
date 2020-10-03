#!/bin/bash
python lowdim.py \
    --qmin 0 \
    --qmax 40 \
    --pmin 0.75 \
    --pmax 1.5 \
    -n 400 \
    "$@"