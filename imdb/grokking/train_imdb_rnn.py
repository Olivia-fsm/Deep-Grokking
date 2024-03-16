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

import matplotlib.pyplot as plt
from tunnel_methods import EarlyExit, Rank

optimizer_dict = {
    'AdamW': torch.optim.AdamW,
    'Adam': torch.optim.Adam,
    'SGD': torch.optim.SGD
}

activation_dict = {
    'ReLU': nn.ReLU,
    'Tanh': nn.Tanh,
    'Sigmoid': nn.Sigmoid,
    'GELU': nn.GELU
}

loss_function_dict = {
    'MSE': nn.MSELoss,
    'CrossEntropy': nn.CrossEntropyLoss
}

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def compute_accuracy(network, dataset, device, N=2000, batch_size=50):
    """Computes accuracy of `network` on `dataset`.
    """
    with torch.no_grad():
        N = min(len(dataset), N)
        batch_size = min(batch_size, N)
        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        correct = 0
        total = 0
        for x, labels in islice(dataset_loader, N // batch_size):
            logits = network(x.to(device))
            predicted_labels = torch.argmax(logits, dim=1)
            correct += torch.sum(predicted_labels == labels.to(device))
            total += x.size(0)
        return (correct / total).item()

def compute_loss(network, dataset, loss_function, device, N=2000, batch_size=50):
    """Computes mean loss of `network` on `dataset`.
    """
    with torch.no_grad():
        N = min(len(dataset), N)
        batch_size = min(batch_size, N)
        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        loss_fn = loss_function_dict[loss_function](reduction='sum')
        one_hots = torch.eye(10, 10).to(device)
        total = 0
        points = 0
        for x, labels in islice(dataset_loader, N // batch_size):
            y = network(x.to(device))
            if loss_function == 'CrossEntropy':
                total += loss_fn(y, labels.to(device)).item()
            elif loss_function == 'MSE':
                total += loss_fn(y, one_hots[labels]).item()
            points += len(labels)
        return total / points

def get_list_of_layers(model, exclude_list=None):
    # filter non leaf modules and ones that are not Convs, BNs or Linear e.g. non-linear activations
    if exclude_list is None:
        exclude_list = []
    layers = [
        n
        for n, m in model.named_modules()
        if len(list(m.children())) == 0
        and (
            isinstance(m, (torch.nn.Linear, torch.nn.Conv2d))
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

# load dataset
def get_data(data_dir:str, train_points=-1):
    train = torchvision.datasets.MNIST(root=data_dir, train=True, 
        transform=torchvision.transforms.ToTensor(), download=True)
    test = torchvision.datasets.MNIST(root=data_dir, train=False, 
        transform=torchvision.transforms.ToTensor(), download=True)
    if train_points>0:
        train = torch.utils.data.Subset(train, range(train_points))
    return train, test

# create model
def get_mlp(depth, width, activation, initialization_scale, device="cuda"):
    assert activation in activation_dict, f"Unsupported activation function: {activation}"
    activation_fn = activation_dict[activation]
    layers = [nn.Flatten()]
    for i in range(depth):
        if i == 0:
            layers.append(nn.Linear(784, width))
            layers.append(activation_fn())
        elif i == depth - 1:
            layers.append(nn.Linear(width, 10))
        else:
            layers.append(nn.Linear(width, width))
            layers.append(activation_fn())
    mlp = nn.Sequential(*layers).to(device)
    with torch.no_grad():
        for p in mlp.parameters():
            p.data = initialization_scale * p.data
    return mlp

def train_mlp_mnist(mlp, 
                    train_data,
                    test_data,
                    batch_size=200,
                    lr=1e-3,
                    weight_decay=0.01,
                    optimization_steps=10000, 
                    optimizer="AdamW",
                    loss_function="MSE",
                    log_freq=None,
                    device="cuda",
                    rank=False,
                    probe=False,
                    use_wandb=True):
    # create optimizer
    assert optimizer in optimizer_dict, f"Unsupported optimizer choice: {optimizer}"
    optimizer = optimizer_dict[optimizer](mlp.parameters(), lr=lr, weight_decay=weight_decay)

    # define loss function
    assert loss_function in loss_function_dict
    loss_fn = loss_function_dict[loss_function]()
    
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    norms = []
    last_layer_norms = []
    log_steps = []
    rpt_dict = {}
    layer_list = get_list_of_layers(model=mlp)
    # rank analysis
    if rank:
        rpt_dict["rank"] = {l:[] for l in layer_list}
    if probe:
        rpt_dict["probe"] = {l:[] for l in layer_list}

    steps = 0
    one_hots = torch.eye(10, 10).to(device)
    
    train = train_data
    test = test_data
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    
    if log_freq is None:
        log_freq = math.ceil(optimization_steps / 150)
    probe_freq = 5*log_freq
    
    with tqdm(total=optimization_steps) as pbar:
        for x, labels in islice(cycle(train_loader), optimization_steps):
            if (steps < 30) or (steps < 150 and steps % 10 == 0) or steps % log_freq == 0:
                train_losses.append(compute_loss(mlp, train, loss_function, device, N=len(train)))
                train_accuracies.append(compute_accuracy(mlp, train, device, N=len(train)))
                test_losses.append(compute_loss(mlp, test, loss_function, device, N=len(test)))
                test_accuracies.append(compute_accuracy(mlp, test, device, N=len(test)))
                log_steps.append(steps)
                
                if use_wandb:
                    wandb_dict = {
                        'train/loss': train_losses[-1],
                        'train/acc': train_accuracies[-1],
                        'test/loss': test_losses[-1],
                        'test/acc': test_accuracies[-1],
                        'log_step': log_steps[-1],
                    }
                    
                # rank analysis
                if rank and (steps % probe_freq == 0):
                    rank_result = rank_analysis(model=mlp,
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
                if probe and (steps % probe_freq == 0):
                    probe_result = early_exits(model=mlp,
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
                with torch.no_grad():
                    total = sum(torch.pow(p, 2).sum() for p in mlp.parameters())
                    norms.append(float(np.sqrt(total.item())))
                    last_layer = sum(torch.pow(p, 2).sum() for p in mlp[-1].parameters())
                    last_layer_norms.append(float(np.sqrt(last_layer.item())))
                    wandb_dict["norm"] = norms[-1]
                    wandb_dict["last_layer_norm"] = last_layer_norms[-1]
                    
                pbar.set_description("L: {0:1.1e}|{1:1.1e}. A: {2:2.1f}%|{3:2.1f}%".format(
                    train_losses[-1],
                    test_losses[-1],
                    train_accuracies[-1] * 100, 
                    test_accuracies[-1] * 100))
                if use_wandb:
                    wandb.log(wandb_dict, commit=True) 

            optimizer.zero_grad()
            mlp.train()
            mlp.to(device)
            y = mlp(x.to(device))
            if loss_function == 'CrossEntropy':
                loss = loss_fn(y, labels.to(device))
            elif loss_function == 'MSE':
                loss = loss_fn(y, one_hots[labels])
            loss.backward()
            optimizer.step()
            steps += 1
            pbar.update(1)
    rpt_dict["train_losses"] = train_losses
    rpt_dict["test_losses"] = test_losses
    rpt_dict["train_accuracies"] = train_accuracies
    rpt_dict["test_accuracies"] = test_accuracies
    rpt_dict["norms"] = norms
    rpt_dict["last_layer_norms"] = last_layer_norms
    rpt_dict["log_steps"] = log_steps
    return mlp, rpt_dict


# Add cli params
import argparse
args_parser = argparse.ArgumentParser()
# DomainConfigArguments
args_parser.add_argument('--data_dir', default='/scratch/homes/sfan/models/Omnigrok/mnist/grokking/MNIST', type=str)
args_parser.add_argument('--wandb_proj', default='mnist-grok', type=str)
args_parser.add_argument('--wandb_run', default='mlp_test', type=str)
args_parser.add_argument('--save_dir', default='/scratch/homes/sfan/models/Omnigrok/mnist/grokking/exp', type=str)
args_parser.add_argument('--train_points', default=1000, type=int)
args_parser.add_argument('--optimization_steps', default=100000, type=int)
args_parser.add_argument('--batch_size', default=200, type=int)
args_parser.add_argument('--loss_function', default="MSE", type=str)
args_parser.add_argument('--optimizer', default="AdamW", type=str)
args_parser.add_argument('--activation', default="ReLU", type=str)
args_parser.add_argument('--weight_decay', default=0.01, type=float)
args_parser.add_argument('--lr', default=1e-3, type=float)
args_parser.add_argument('--initialization_scale', default=10.0, type=float)
args_parser.add_argument('--depth', default=12, type=int)
args_parser.add_argument('--width', default=800, type=int)
args_parser.add_argument('--seed', default=0, type=int)
args_parser.add_argument('--device', default="cuda", type=str)
args_parser.add_argument('--rank', action='store_true')
args_parser.add_argument('--probe', action='store_true')

def run():
    args = args_parser.parse_args()
    os.environ["WANDB_PROJECT"] = args.wandb_proj # name your W&B project 
    wandb.init(project=args.wandb_proj, name=args.wandb_run, config=args)
    
    set_seed(seed=args.seed)
    train, test = get_data(data_dir=args.data_dir, train_points=args.train_points)
    mlp = get_mlp(args.depth, args.width, args.activation, args.initialization_scale, device=args.device)

    mlp, rpt_dict = train_mlp_mnist(mlp, 
                    train_data=train,
                    test_data=test,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    optimization_steps=args.optimization_steps, 
                    optimizer=args.optimizer,
                    loss_function=args.loss_function,
                    log_freq=None,
                    device=args.device,
                    rank=args.rank,
                    probe=args.probe,
                    use_wandb=True,)
    save_exp_dir = os.path.join(args.save_dir, args.wandb_run)
    os.makedirs(save_exp_dir, exist_ok=True)
    torch.save(mlp.state_dict(), os.path.join(save_exp_dir, "mlp.pt"))
    with open(os.path.join(save_exp_dir, "rpt.pkl"), "wb") as trg:
        pickle.dump(rpt_dict, trg)

if __name__ == "__main__":
    run()