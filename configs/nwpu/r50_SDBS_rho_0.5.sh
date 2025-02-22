#!/usr/bin/env bash


python -u main.py \
    --with_box_refine \
    --two_stage \
    --eff_query_init \
    --eff_specific_head \
    --rho 0.5 \
    --use_enc_aux_loss \
     --nproc_per_node=4 \
    --scale_type 1\
    --epochs 1500 \
    --lr_step 1200\
    --scale_p 0.3\
    --lr 1e-4 \
    --dataset nwpu \
    --crop_size 256 \
    --test_per_epoch 20 \
    --batch_size 16 \
    --gpu_id 0
