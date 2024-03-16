#!/bin/bash
cd /scratch/homes/sfan/models/Omnigrok/imdb/grokking
# pip install -r requirements.txt

# export WANDB_API_KEY="put your authorize key here, to find it: https://wandb.ai/authorize"
export WANDB_API_KEY="1dba8bb7f1589f867fa1538683d77eaf4e8209de"

python imdb-grokking.py --wandb_run rnn_d8_tp2000_scale8_wd001 --depth 8 --train_points 2000 --initialization_scale 8 --rank --weight_decay 0.01 --device cuda:3
python imdb-grokking.py --wandb_run rnn_d8_tp2000_scale8_wd10 --depth 8 --train_points 2000 --initialization_scale 8 --rank --weight_decay 1.0 --device cuda:3

# python imdb-grokking.py --wandb_run rnn_d8_tp2000_scale12_wd05 --depth 4 --train_points 1000 --initialization_scale 12.0 --rank --weight_decay 0.5 --device cuda:3

# python imdb-grokking.py --wandb_run rnn_d6_tp1000_scale6_wd20 --depth 6 --train_points 1000 --initialization_scale 6.0 --rank --weight_decay 2.0
# python imdb-grokking.py --wandb_run rnn_d6_tp2000_scale6_wd20 --depth 6 --train_points 2000 --initialization_scale 6.0 --rank --weight_decay 2.0
# python imdb-grokking.py --wandb_run rnn_d6_tp4000_scale6_wd20 --depth 6 --train_points 4000 --initialization_scale 6.0 --rank --weight_decay 2.0