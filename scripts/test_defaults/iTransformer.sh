#!/bin/bash
mode="test"
modelname="iTransformer"

python -u run.py \
    --mode $mode \
    --seed 42 \
    --model $modelname \
    --d_model 256 \
    --d_ff 512 \
    --n_blocks 4 \
    --activation gelu \
    --dropout 0.2 \
    --lookback_window 120 \
    --stride 120 \
    --batch_size 256 \
    --optimizer AdamW
