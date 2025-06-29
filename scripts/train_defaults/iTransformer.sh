#!/bin/bash
mode="train"
modelname="iTransformer"

python -u run.py \
    --mode $mode \
    --seed 42 \
    --model $modelname \
    --n_heads 4 \
    --d_model 256 \
    --d_ff 512 \
    --n_blocks 4 \
    --activation gelu \
    --dropout 0.2 \
    --attn_dropout 0.2 \
    --lookback_window 120 \
    --stride 120 \
    --batch_size 64 \
    --learning_rate 5e-5 \
    --optimizer AdamW \
    --use_early_stop \
    --early_stop_patience 100 \
    --grad_clip \
    --grad_norm 1.0
