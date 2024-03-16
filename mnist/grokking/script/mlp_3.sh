#!/bin/bash
cd /scratch/homes/sfan/models/Omnigrok/mnist/grokking
pip install -r requirements.txt

# export WANDB_API_KEY="put your authorize key here, to find it: https://wandb.ai/authorize"
export WANDB_API_KEY="1dba8bb7f1589f867fa1538683d77eaf4e8209de"

python train_mnist_mlp.py --wandb_run mlp_d12_w400_tp2000_scale10_wd001 --depth 12 --width 400 --train_points 2000 --optimization_steps 100000 --initialization_scale 10.0 --rank --probe --weight_decay 0.01 --device cuda:0
python train_mnist_mlp.py --wandb_run mlp_d12_w400_tp2000_scale6_wd001 --depth 12 --width 400 --train_points 2000 --optimization_steps 100000 --initialization_scale 6.0 --rank --probe --weight_decay 0.01 --device cuda:0
python train_mnist_mlp.py --wandb_run mlp_d12_w400_tp2000_scale4_wd001 --depth 12 --width 400 --train_points 2000 --optimization_steps 100000 --initialization_scale 4.0 --rank --probe --weight_decay 0.01 --device cuda:0
python train_mnist_mlp.py --wandb_run mlp_d12_w400_tp2000_scale2_wd001 --depth 12 --width 400 --train_points 2000 --optimization_steps 100000 --initialization_scale 2.0 --rank --probe --weight_decay 0.01 --device cuda:0

# python train_mnist_mlp.py --wandb_run mlp_d8_w400_tp2000_scale8_wd005 --depth 8 --width 400 --train_points 2000 --optimization_steps 100000 --initialization_scale 8.0 --rank --probe --weight_decay 0.05 --device cuda:0
# python train_mnist_mlp.py --wandb_run mlp_d8_w400_tp2000_scale8_wd001 --depth 8 --width 400 --train_points 2000 --optimization_steps 100000 --initialization_scale 8.0 --rank --probe --weight_decay 0.01 --device cuda:0
# python train_mnist_mlp.py --wandb_run mlp_d8_w400_tp2000_scale8_wd0005 --depth 8 --width 400 --train_points 2000 --optimization_steps 100000 --initialization_scale 8.0 --rank --probe --weight_decay 0.005 --device cuda:0
# python train_mnist_mlp.py --wandb_run mlp_d8_w400_tp2000_scale8_wd01 --depth 8 --width 400 --train_points 2000 --optimization_steps 100000 --initialization_scale 8.0 --rank --probe --weight_decay 0.1 --device cuda:0

# python train_mnist_mlp.py --wandb_run mlp_d4_w400_tp2000_scale8_wd005 --depth 4 --width 400 --train_points 2000 --optimization_steps 100000 --initialization_scale 8.0 --rank --probe --weight_decay 0.05 --device cuda:0
# python train_mnist_mlp.py --wandb_run mlp_d4_w400_tp2000_scale8_wd001 --depth 4 --width 400 --train_points 2000 --optimization_steps 100000 --initialization_scale 8.0 --rank --probe --weight_decay 0.01 --device cuda:0
# python train_mnist_mlp.py --wandb_run mlp_d4_w400_tp2000_scale8_wd0005 --depth 4 --width 400 --train_points 2000 --optimization_steps 100000 --initialization_scale 8.0 --rank --probe --weight_decay 0.005 --device cuda:0
# python train_mnist_mlp.py --wandb_run mlp_d4_w400_tp2000_scale8_wd01 --depth 4 --width 400 --train_points 2000 --optimization_steps 100000 --initialization_scale 8.0 --rank --probe --weight_decay 0.1 --device cuda:0


# python train_mnist_mlp.py --wandb_run mlp_d8_w400_tp2000_scale8_wd0005 --depth 8 --width 400 --train_points 2000 --optimization_steps 100000 --initialization_scale 8.0 --rank --probe --weight_decay 0.005
# python train_mnist_mlp.py --wandb_run mlp_d8_w400_tp2000_scale8_wd01 --depth 8 --width 400 --train_points 2000 --optimization_steps 100000 --initialization_scale 8.0 --rank --probe --weight_decay 0.1

# python train_mnist_mlp.py --wandb_run mlp_d4_w400_tp2000_scale8_wd0005 --depth 8 --width 400 --train_points 2000 --optimization_steps 100000 --initialization_scale 8.0 --rank --probe --weight_decay 0.005
# python train_mnist_mlp.py --wandb_run mlp_d4_w400_tp2000_scale8_wd01 --depth 8 --width 400 --train_points 2000 --optimization_steps 100000 --initialization_scale 8.0 --rank --probe --weight_decay 0.1
