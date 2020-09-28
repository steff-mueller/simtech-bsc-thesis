#!/bin/bash
python lowdim.py \
    --qmin -10 \
    --qmax 10 \
    --pmin -3 \
    --pmax 3 \
    "$@"