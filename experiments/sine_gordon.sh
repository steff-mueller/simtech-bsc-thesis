#!/bin/bash
python wave.py \
    --architecture nonlinear \
    --model sine_gordon \
    --dt 0.01 \
    --nx 4000 \
    -l 50 \
    --epochs 2000 \
    "$@"