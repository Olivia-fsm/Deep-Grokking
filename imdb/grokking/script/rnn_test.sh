#!/bin/bash
cd /scratch/homes/sfan/models/Omnigrok/imdb/grokking
# pip install -r requirements.txt

# export WANDB_API_KEY="put your authorize key here, to find it: https://wandb.ai/authorize"
export WANDB_API_KEY="1dba8bb7f1589f867fa1538683d77eaf4e8209de"

python imdb-grokking.py --wandb_run rnn_d4_tp2000_scale4_wd05_sharpness --depth 4 --train_points 2000 --initialization_scale 4.0 --rank --weight_decay 0.5
# python imdb-grokking.py --wandb_run rnn_d6_tp2000_scale2_wd20 --depth 6 --train_points 2000 --initialization_scale 2.0 --rank --weight_decay 2.0
# python imdb-grokking.py --wandb_run rnn_d6_tp2000_scale10_wd20 --depth 6 --train_points 2000 --initialization_scale 10.0 --rank --weight_decay 2.0