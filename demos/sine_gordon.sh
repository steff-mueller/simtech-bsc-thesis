#!/bin/bash
python wave.py \
    --architecture nonlinear \
    --model sine_gordon \
    --dt 0.0125 \
    --nx 2000 \
    -l 50 \
    --epochs 2000 \
    "$@"