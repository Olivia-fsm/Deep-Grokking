#!/bin/bash
cd /scratch/homes/sfan/models/Omnigrok/imdb/grokking
# pip install -r requirements.txt

# export WANDB_API_KEY="put your authorize key here, to find it: https://wandb.ai/authorize"
export WANDB_API_KEY="1dba8bb7f1589f867fa1538683d77eaf4e8209de"

python imdb-grokking.py --wandb_run rnn_d4_tp4000_scale8_wd01 --depth 4 --train_points 4000 --initialization_scale 8.0 --rank --weight_decay 0.1 --device cuda:2
python imdb-grokking.py --wandb_run rnn_d4_tp4000_scale8_wd001 --depth 4 --train_points 4000 --initialization_scale 8.0 --rank --weight_decay 0.01 --device cuda:2
# python imdb-grokking.py --wandb_run rnn_d4_tp4000_scale8_wd25 --depth 4 --train_points 4000 --initialization_scale 8.0 --rank --weight_decay 2.5 --device cuda:1

# python imdb-grokking.py --wandb_run rnn_d4_tp1000_scale8_wd25 --depth 4 --train_points 1000 --initialization_scale 8.0 --rank --weight_decay 2.5 --device cuda:2
# python imdb-grokking.py --wandb_run rnn_d4_tp1000_scale8_wd05 --depth 4 --train_points 1000 --initialization_scale 8.0 --rank --weight_decay 0.5 --device cuda:2
# python imdb-grokking.py --wandb_run rnn_d4_tp1000_scale6_wd10 --depth 4 --train_points 1000 --initialization_scale 6.0 --rank --weight_decay 1.0 --device cuda:1