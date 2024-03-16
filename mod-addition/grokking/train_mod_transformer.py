from collections import defaultdict
from itertools import islice
import random
import time
from pathlib import Path
import math

import os
import wandb
import numpy as np
from tqdm import tqdm
import pickle

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

import matplotlib.pyplot as plt
from tunnel_methods import EarlyExit, Rank

import einops


# This code was taken directly from Neel Nanda's study of grokking:
# https://colab.research.google.com/drive/1F6_1_cWXE5M7WocUcpQWp3v8z4b1jL20

class HookPoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.fwd_hooks = []
        self.bwd_hooks = []
    
    def give_name(self, name):
        # Called by the model at initialisation
        self.name = name
    
    def add_hook(self, hook, dir='fwd'):
        # Hook format is fn(activation, hook_name)
        # Change it into PyTorch hook format (this includes input and output, 
        # which are the same for a HookPoint)
        def full_hook(module, module_input, module_output):
            return hook(module_output, name=self.name)
        if dir=='fwd':
            handle = self.register_forward_hook(full_hook)
            self.fwd_hooks.append(handle)
        elif dir=='bwd':
            handle = self.register_backward_hook(full_hook)
            self.bwd_hooks.append(handle)
        else:
            raise ValueError(f"Invalid direction {dir}")
    
    def remove_hooks(self, dir='fwd'):
        if (dir=='fwd') or (dir=='both'):
            for hook in self.fwd_hooks:
                hook.remove()
            self.fwd_hooks = []
        if (dir=='bwd') or (dir=='both'):
            for hook in self.bwd_hooks:
                hook.remove()
            self.bwd_hooks = []
        if dir not in ['fwd', 'bwd', 'both']:
            raise ValueError(f"Invalid direction {dir}")
    
    def forward(self, x):
        return x

# =========== Model ============= #
# Embed & Unembed
class Embed(nn.Module):
    def __init__(self, d_vocab, d_model):
        super().__init__()
        self.W_E = nn.Parameter(torch.randn(d_model, d_vocab)/np.sqrt(d_model))
    
    def forward(self, x):
        return torch.einsum('dbp -> bpd', self.W_E[:, x])

class Unembed(nn.Module):
    def __init__(self, d_vocab, d_model):
        super().__init__()
        self.W_U = nn.Parameter(torch.randn(d_model, d_vocab)/np.sqrt(d_vocab))
    
    def forward(self, x):
        return (x @ self.W_U)

# Positional Embeddings
class PosEmbed(nn.Module):
    def __init__(self, max_ctx, d_model):
        super().__init__()
        self.W_pos = nn.Parameter(torch.randn(max_ctx, d_model)/np.sqrt(d_model))
    
    def forward(self, x):
        return x+self.W_pos[:x.shape[-2]]

# LayerNorm
class LayerNorm(nn.Module):
    def __init__(self, d_model, epsilon = 1e-4, model=[None]):
        super().__init__()
        self.model = model
        self.w_ln = nn.Parameter(torch.ones(d_model))
        self.b_ln = nn.Parameter(torch.zeros(d_model))
        self.epsilon = epsilon
    
    def forward(self, x):
        if self.model[0].use_ln:
            x = x - x.mean(axis=-1)[..., None]
            x = x / (x.std(axis=-1)[..., None] + self.epsilon)
            x = x * self.w_ln
            x = x + self.b_ln
            return x
        else:
            return x

# Attention
class Attention(nn.Module):
    def __init__(self, d_model, num_heads, d_head, n_ctx, model):
        super().__init__()
        self.model = model
        self.W_K = nn.Parameter(torch.randn(num_heads, d_head, d_model)/np.sqrt(d_model))
        self.W_Q = nn.Parameter(torch.randn(num_heads, d_head, d_model)/np.sqrt(d_model))
        self.W_V = nn.Parameter(torch.randn(num_heads, d_head, d_model)/np.sqrt(d_model))
        self.W_O = nn.Parameter(torch.randn(d_model, d_head * num_heads)/np.sqrt(d_model))
        self.register_buffer('mask', torch.tril(torch.ones((n_ctx, n_ctx))))
        self.d_head = d_head
        self.hook_k = HookPoint()
        self.hook_q = HookPoint()
        self.hook_v = HookPoint()
        self.hook_z = HookPoint()
        self.hook_attn = HookPoint()
        self.hook_attn_pre = HookPoint()

    def forward(self, x):
        k = self.hook_k(torch.einsum('ihd,bpd->biph', self.W_K, x))
        q = self.hook_q(torch.einsum('ihd,bpd->biph', self.W_Q, x))
        v = self.hook_v(torch.einsum('ihd,bpd->biph', self.W_V, x))
        attn_scores_pre = torch.einsum('biph,biqh->biqp', k, q)
        attn_scores_masked = torch.tril(attn_scores_pre) - 1e10 * (1 - self.mask[:x.shape[-2], :x.shape[-2]])
        attn_matrix = self.hook_attn(F.softmax(self.hook_attn_pre(attn_scores_masked/np.sqrt(self.d_head)), dim=-1))
        z = self.hook_z(torch.einsum('biph,biqp->biqh', v, attn_matrix))
        z_flat = einops.rearrange(z, 'b i q h -> b q (i h)')
        out = torch.einsum('df,bqf->bqd', self.W_O, z_flat)
        return out

# MLP Layers
class MLP(nn.Module):
    def __init__(self, d_model, d_mlp, act_type, model):
        super().__init__()
        self.model = model
        self.W_in = nn.Parameter(torch.randn(d_mlp, d_model)/np.sqrt(d_model))
        self.b_in = nn.Parameter(torch.zeros(d_mlp))
        self.W_out = nn.Parameter(torch.randn(d_model, d_mlp)/np.sqrt(d_model))
        self.b_out = nn.Parameter(torch.zeros(d_model))
        self.act_type = act_type
        # self.ln = LayerNorm(d_mlp, model=self.model)
        self.hook_pre = HookPoint()
        self.hook_post = HookPoint()
        assert act_type in ['ReLU', 'GeLU']
        
    def forward(self, x):
        x = self.hook_pre(torch.einsum('md,bpd->bpm', self.W_in, x) + self.b_in)
        if self.act_type=='ReLU':
            x = F.relu(x)
        elif self.act_type=='GeLU':
            x = F.gelu(x)
        x = self.hook_post(x)
        x = torch.einsum('dm,bpm->bpd', self.W_out, x) + self.b_out
        return x

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_mlp, d_head, num_heads, n_ctx, act_type, model):
        super().__init__()
        self.model = model
        # self.ln1 = LayerNorm(d_model, model=self.model)
        self.attn = Attention(d_model, num_heads, d_head, n_ctx, model=self.model)
        # self.ln2 = LayerNorm(d_model, model=self.model)
        self.mlp = MLP(d_model, d_mlp, act_type, model=self.model)
        self.hook_attn_out = HookPoint()
        self.hook_mlp_out = HookPoint()
        self.hook_resid_pre = HookPoint()
        self.hook_resid_mid = HookPoint()
        self.hook_resid_post = HookPoint()
    
    def forward(self, x):
        x = self.hook_resid_mid(x + self.hook_attn_out(self.attn((self.hook_resid_pre(x)))))
        x = self.hook_resid_post(x + self.hook_mlp_out(self.mlp((x))))
        return x

# Full transformer
class Transformer(nn.Module):
    def __init__(self, num_layers, d_vocab, d_model, d_mlp, d_head, num_heads, n_ctx, act_type, use_cache=False, use_ln=True):
        super().__init__()
        self.cache = {}
        self.use_cache = use_cache

        self.embed = Embed(d_vocab, d_model)
        self.pos_embed = PosEmbed(n_ctx, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, d_mlp, d_head, num_heads, n_ctx, act_type, model=[self]) for i in range(num_layers)])
        # self.ln = LayerNorm(d_model, model=[self])
        self.unembed = Unembed(d_vocab, d_model)
        self.use_ln = use_ln

        for name, module in self.named_modules():
            if type(module)==HookPoint:
                module.give_name(name)
    
    def forward(self, x):
        x = self.embed(x)
        x = self.pos_embed(x)
        for block in self.blocks:
            x = block(x)
        # x = self.ln(x)
        x = self.unembed(x)
        return x

    def set_use_cache(self, use_cache):
        self.use_cache = use_cache
    
    def hook_points(self):
        return [module for name, module in self.named_modules() if 'hook' in name]

    def remove_all_hooks(self):
        for hp in self.hook_points():
            hp.remove_hooks('fwd')
            hp.remove_hooks('bwd')
    
    def cache_all(self, cache, incl_bwd=False):
        # Caches all activations wrapped in a HookPoint
        def save_hook(tensor, name):
            cache[name] = tensor.detach()
        def save_hook_back(tensor, name):
            cache[name+'_grad'] = tensor[0].detach()
        for hp in self.hook_points():
            hp.add_hook(save_hook, 'fwd')
            if incl_bwd:
                hp.add_hook(save_hook_back, 'bwd')

# =================== End Model =================== #

# =================== Metric =================== #
def full_loss(model, data, device):
    loader = torch.utils.data.DataLoader(data, batch_size=len(data), shuffle=False)
    # Take the final position only
    x, labels = next(iter(loader))
    x = x.to(device)
    labels = labels.to(device)
    logits = model(x)[:, -1]
    return torch.nn.functional.cross_entropy(logits, labels)

def full_accuracy(model, data, device):
    loader = torch.utils.data.DataLoader(data, batch_size=len(data), shuffle=False)
    # Take the final position only
    x, labels = next(iter(loader))
    x = x.to(device)
    labels = labels.to(device)
    logits = model(x)[:, -1]
    predictions = torch.argmax(logits, dim=1)
    return torch.sum(predictions == labels).item() / len(labels)

# =================== End Metric =================== #

# def get_list_of_layers(model, exclude_list=None):
#     # filter non leaf modules and ones that are not Convs, BNs or Linear e.g. non-linear activations
#     if exclude_list is None:
#         exclude_list = []
#     layers = [
#         n
#         for n, m in model.named_modules()
#         if len(list(m.children())) == 0
#         and (
#             isinstance(m, (torch.nn.Linear, torch.nn.Conv2d))
#         )  # , torch.nn.BatchNorm2d)))
#     ]
#     # filter out modules that should be excluded
#     layers = [n for n in layers if not sum([j in n.lower() for j in exclude_list])]
#     return layers

def get_list_of_layers(model, exclude_list=None):
    # filter non leaf modules and ones that are not Convs, BNs or Linear e.g. non-linear activations
    if exclude_list is None:
        exclude_list = []
    layers = [
        n
        for n, m in model.named_modules()
        if (
            n.endswith("mlp")
            # isinstance(m, (torch.nn.Linear, torch.nn.Conv2d))
        )  # , torch.nn.BatchNorm2d)))
    ]
    # filter out modules that should be excluded
    layers = [n for n in layers if not sum([j in n.lower() for j in exclude_list])]
    return layers

def rank_analysis(
    model, 
    dataset, 
    layers: list, 
    name=None,
    rpath=None, 
):
    rank = Rank(
        model=model, data=dataset, layers=layers, rpath=rpath, plotter=lambda x: x + 1
    )
    rank.analysis()
    if rpath is not None:
        rank.export(name)
        # rank.plot(name)
    rank.clean_up()
    return rank.result

def early_exits(
    model,
    train_data,
    test_data,
    layers: list,
    name=None,
    rpath=None,
    verbose=False
):
    early_exit = EarlyExit(
        model=model,
        train_data=train_data,  # type: ignore
        test_data=test_data,
        layers=layers,
        rpath=rpath,
        plotter=lambda x: x + 1,
    )

    early_exit.analysis(verbose=verbose)
    # early_exit.analysis()
    if rpath is not None:
        early_exit.export(name=name)
        # early_exit.plot(name=name)
    early_exit.clean_up()
    return early_exit.result

def set_seed(seed=0, dtype=torch.float64):
    torch.set_default_dtype(dtype)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def train_transformer_mod(p,
                          alpha,
                          train_fraction,
                          num_layers=2,
                          num_heads=4,
                          n_ctx=3,
                          lr=1e-3, 
                          weight_decay=1.0,
                          opt_steps=15000,
                          device="cuda:0",
                          rank=False,
                          probe=False,
                          use_wandb=True):
    equals_token = p
    x, y = torch.meshgrid(torch.arange(p), torch.arange(p), indexing='ij')
    x = x.flatten()
    y = y.flatten()
    # plus = torch.ones(x.shape, dtype=torch.int64) * plus_token
    equals = torch.ones(x.shape, dtype=torch.int64) * equals_token
    prompts = torch.stack([x, y, equals], dim=1).to(device)
    answers = ((x + y) % p).to(device)

    data = torch.utils.data.TensorDataset(prompts, answers)
    train, test = torch.utils.data.random_split(data, 
                                    [int(train_fraction * len(data)),
                                    len(data) - int(train_fraction * len(data))
                                    ])

    model = Transformer(num_layers=num_layers, 
                        d_vocab=equals_token+1, 
                        d_model=128,
                        d_mlp=512,
                        d_head=32,
                        num_heads=num_heads,
                        n_ctx=3, # context length
                        act_type='ReLU', 
                        use_cache=False, 
                        use_ln=False # use LayerNorm
                    ).to(device)
    # TODO: add init scale
    with torch.no_grad():
        for param in model.parameters():
            param.data *= alpha
    
    layer_list = get_list_of_layers(model=model)
    # rank analysis
    rpt_dict = {}
    if rank:
        rpt_dict["rank"] = {l:[] for l in layer_list}
    if probe:
        rpt_dict["probe"] = {l:[] for l in layer_list}
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98))
    log_steps = []
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    norms = []
    for step in tqdm(range(opt_steps)):
        train_loss = full_loss(model, train, device)

        if step % 10 == 0:
            with torch.no_grad():
                log_steps.append(step)
                test_loss = full_loss(model, test, device)
                train_losses.append(train_loss.item())
                test_losses.append(test_loss.item())
                train_accuracies.append(full_accuracy(model, train, device))
                test_accuracies.append(full_accuracy(model, test, device))
                norms.append(np.sqrt(sum(param.pow(2).sum().item() for param in model.parameters())))
                if use_wandb:
                    wandb_dict = {
                        'train/loss': train_losses[-1],
                        'train/acc': train_accuracies[-1],
                        'test/loss': test_losses[-1],
                        'test/acc': test_accuracies[-1],
                        'log_step': log_steps[-1],
                        'norm': norms[-1]
                    }
                # rank analysis
                if rank:
                    rank_result = rank_analysis(model=model,
                                                dataset=test,
                                                layers=layer_list,
                                                name="rank",
                                                rpath=None,)
                    rank_result = rank_result['rank']
                    for i,v in rank_result.items():
                        rpt_dict["rank"][i].append(v)
                    if use_wandb:
                        for l,v in rank_result.items():
                            wandb_dict[f"rank/layer_{l}"] = v
                if probe:
                    probe_result = early_exits(model=model,
                                               train_data=train,
                                               test_data=test,
                                               layers=layer_list,
                                               name="probe",
                                               rpath=None,)
                    for i,v in probe_result.items():
                        rpt_dict["probe"][i].append(v)
                    if use_wandb:
                        for l,v in probe_result.items():
                            wandb_dict[f"probe_acc/layer_{l}"] = v
                if use_wandb:
                    wandb.log(wandb_dict, commit=True) 
        model = model.to(device)
        model.train()
        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    rpt_dict["train_losses"] = train_losses
    rpt_dict["test_losses"] = test_losses
    rpt_dict["train_accuracies"] = train_accuracies
    rpt_dict["test_accuracies"] = test_accuracies
    rpt_dict["norms"] = norms
    rpt_dict["log_steps"] = log_steps
    return model, rpt_dict

# Add cli params
import argparse
args_parser = argparse.ArgumentParser()

args_parser.add_argument('--data_dir', default='/scratch/homes/sfan/models/deep-grokking/mod-addition/grokking/MNIST', type=str)
args_parser.add_argument('--wandb_proj', default='transformer_grok', type=str)
args_parser.add_argument('--wandb_run', default='mlp_test', type=str)
args_parser.add_argument('--save_dir', default='/scratch/homes/sfan/models/deep-grokking/mod-addition/grokking/exp', type=str)
args_parser.add_argument('--seed', default=42, type=int)
args_parser.add_argument('--p', default=113, type=int)
args_parser.add_argument('--train_fraction', default=0.3, type=float)

args_parser.add_argument('--opt_steps', default=30000, type=int)
args_parser.add_argument('--weight_decay', default=0.01, type=float)
args_parser.add_argument('--lr', default=1e-4, type=float)
args_parser.add_argument('--initialization_scale', default=1.0, type=float)
args_parser.add_argument('--num_layers', default=6, type=int)
args_parser.add_argument('--num_heads', default=4, type=int)

args_parser.add_argument('--device', default="cuda", type=str)
args_parser.add_argument('--rank', action='store_true')
args_parser.add_argument('--probe', action='store_true')

def run():
    args = args_parser.parse_args()
    os.environ["WANDB_PROJECT"] = args.wandb_proj # name your W&B project 
    wandb.init(project=args.wandb_proj, name=args.wandb_run, config=args)
    set_seed(seed=args.seed)
    
    transformer_model, rpt_dict = train_transformer_mod(
                                    p=args.p,
                                    alpha=args.initialization_scale,
                                    train_fraction=args.train_fraction,
                                    num_layers=args.num_layers,
                                    num_heads=args.num_heads,
                                    n_ctx=3,
                                    lr=args.lr, 
                                    weight_decay=args.weight_decay,
                                    opt_steps=args.opt_steps,
                                    device=args.device,
                                    rank=args.rank,
                                    probe=args.probe,
                                    use_wandb=True,)
    save_exp_dir = os.path.join(args.save_dir, args.wandb_run)
    os.makedirs(save_exp_dir, exist_ok=True)
    torch.save(transformer_model.state_dict(), os.path.join(save_exp_dir, "transformer_model.pt"))
    with open(os.path.join(save_exp_dir, "rpt.pkl"), "wb") as trg:
        pickle.dump(rpt_dict, trg)

if __name__ == "__main__":
    run()