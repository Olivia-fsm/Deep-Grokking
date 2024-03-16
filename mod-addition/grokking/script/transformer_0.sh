#!/bin/bash
cd /scratch/homes/sfan/models/deep-grokking/mod-addition/grokking
pip install -r requirements.txt

# export WANDB_API_KEY="put your authorize key here, to find it: https://wandb.ai/authorize"
export WANDB_API_KEY="1dba8bb7f1589f867fa1538683d77eaf4e8209de"

python train_mod_transformer.py --wandb_run transformer_d1_frac03_scale1_wd10 --num_layers 1 --num_heads 4 --initialization_scale 1.0 --rank --probe --weight_decay 1.0 --device cuda:0
python train_mod_transformer.py --wandb_run transformer_d3_frac03_scale1_wd10 --num_layers 3 --num_heads 4 --initialization_scale 1.0 --rank --probe --weight_decay 1.0 --device cuda:0
python train_mod_transformer.py --wandb_run transformer_d6_frac03_scale1_wd10 --num_layers 6 --num_heads 4 --initialization_scale 1.0 --rank --probe --weight_decay 1.0 --device cuda:0

python train_mod_transformer.py --wandb_run transformer_d1_frac03_scale05_wd10 --num_layers 1 --num_heads 4 --initialization_scale 0.5 --rank --probe --weight_decay 1.0 --device cuda:0
python train_mod_transformer.py --wandb_run transformer_d3_frac03_scale05_wd10 --num_layers 3 --num_heads 4 --initialization_scale 0.5 --rank --probe --weight_decay 1.0 --device cuda:0
python train_mod_transformer.py --wandb_run transformer_d6_frac03_scale05_wd10 --num_layers 6 --num_heads 4 --initialization_scale 0.5 --rank --probe --weight_decay 1.0 --device cuda:0

python train_mod_transformer.py --wandb_run transformer_d1_frac05_scale1_wd10 --num_layers 1 --num_heads 4 --initialization_scale 1.0 --rank --probe --weight_decay 1.0 --train_fraction 0.5 --device cuda:0
python train_mod_transformer.py --wandb_run transformer_d3_frac05_scale1_wd10 --num_layers 3 --num_heads 4 --initialization_scale 1.0 --rank --probe --weight_decay 1.0 --train_fraction 0.5 --device cuda:0
python train_mod_transformer.py --wandb_run transformer_d6_frac05_scale1_wd10 --num_layers 6 --num_heads 4 --initialization_scale 1.0 --rank --probe --weight_decay 1.0 --train_fraction 0.5 --device cuda:0

python train_mod_transformer.py --wandb_run transformer_d1_frac05_scale05_wd10 --num_layers 1 --num_heads 4 --initialization_scale 0.5 --rank --probe --weight_decay 1.0 --train_fraction 0.5 --device cuda:0
python train_mod_transformer.py --wandb_run transformer_d3_frac05_scale05_wd10 --num_layers 3 --num_heads 4 --initialization_scale 0.5 --rank --probe --weight_decay 1.0 --train_fraction 0.5 --device cuda:0
python train_mod_transformer.py --wandb_run transformer_d6_frac05_scale05_wd10 --num_layers 6 --num_heads 4 --initialization_scale 0.5 --rank --probe --weight_decay 1.0 --train_fraction 0.5 --device cuda:0

python train_mod_transformer.py --wandb_run transformer_d1_frac05_scale05_wd05 --num_layers 1 --num_heads 4 --initialization_scale 0.5 --rank --probe --weight_decay 0.5 --train_fraction 0.5 --device cuda:0
python train_mod_transformer.py --wandb_run transformer_d3_frac05_scale05_wd05 --num_layers 3 --num_heads 4 --initialization_scale 0.5 --rank --probe --weight_decay 0.5 --train_fraction 0.5 --device cuda:0
python train_mod_transformer.py --wandb_run transformer_d6_frac05_scale05_wd05 --num_layers 6 --num_heads 4 --initialization_scale 0.5 --rank --probe --weight_decay 0.5 --train_fraction 0.5 --device cuda:0

python train_mod_transformer.py --wandb_run transformer_d1_frac05_scale05_wd20 --num_layers 1 --num_heads 4 --initialization_scale 0.5 --rank --probe --weight_decay 2.0 --train_fraction 0.5 --device cuda:0
python train_mod_transformer.py --wandb_run transformer_d3_frac05_scale05_wd20 --num_layers 3 --num_heads 4 --initialization_scale 0.5 --rank --probe --weight_decay 2.0 --train_fraction 0.5 --device cuda:0
python train_mod_transformer.py --wandb_run transformer_d6_frac05_scale05_wd20 --num_layers 6 --num_heads 4 --initialization_scale 0.5 --rank --probe --weight_decay 2.0 --train_fraction 0.5 --device cuda:0

# python train_mod_transformer.py --wandb_run mlp_d12_w400_tp2000_scale8_wd0015 --depth 12 --width 400 --train_points 2000 --optimization_steps 100000 --initialization_scale 8.0 --rank --probe --weight_decay 0.015 --device cuda:0
# python train_mod_transformer.py --wandb_run mlp_d4_w400_tp2000_scale8_wd002 --depth 4 --width 400 --train_points 2000 --optimization_steps 100000 --initialization_scale 8.0 --rank --probe --weight_decay 0.02 --device cuda:0
# python train_mod_transformer.py --wandb_run mlp_d4_w400_tp2000_scale8_wd0015 --depth 4 --width 400 --train_points 2000 --optimization_steps 100000 --initialization_scale 8.0 --rank --probe --weight_decay 0.015 --device cuda:0# python train_mnist_mlp.py --wandb_run mlp_d12_w400_tp2000_scale8_wd0005 --depth 12 --width 400 --train_points 2000 --optimization_steps 100000 --initialization_scale 8.0 --rank --probe --weight_decay 0.005 --device cuda:0
