import os
from abc import ABC, abstractmethod
from functools import partial
from typing import Callable
import copy

# from loguru import logger
import torch
from torch.linalg import matrix_rank
import torch.nn as nn
import numpy as np
from torch.utils.data import random_split, DataLoader, TensorDataset
from compute_hessian import hessian

Dataset = torch.utils.data.Dataset
Model = torch.nn.Module
Tensor = torch.Tensor

def get_list_of_layers(model, exclude_list):
    # filter non leaf modules and ones that are not Convs, BNs or Linear e.g. non-linear activations
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

class LinearProbe(nn.Module):
    def __init__(
        self, feature_size: int, task_size: int, epochs: int = 30, lr: float = 1e-3
    ):
        """Train a linear layer (linear probe - LP ) on extracted representations.

        Args:
            feature_size (int): size of flattened vector of representations
            task_size (int): number of classes in task
            epochs (int, optional): number of epochs to train LP. Defaults to 30.
            lr (float, optional): learning rate to train LP. Defaults to 1e-3.
        """
        super().__init__()
        self.linear_probe = nn.Linear(feature_size, task_size)
        self.optimizer = torch.optim.Adam(self.linear_probe.parameters(), lr=lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.epochs = epochs
        self.best_val_acc = 0.0
        self.cuda = torch.cuda.is_available()  # type: ignore
        if self.cuda:
            self.linear_probe = self.linear_probe.cuda()

    def forward(self, x):
        return self.linear_probe(x)

    @staticmethod
    def create_loaders(dataset, split_ratio=(0.8, 0.2)):
        X, y = dataset
        dataset = TensorDataset(X, y)
        datasets = random_split(dataset, split_ratio)
        loader = partial(DataLoader, batch_size=512, shuffle=True, num_workers=0)
        loaders = [loader(dataset) for dataset in datasets]
        return loaders

    def train(self, X, y, verbose=False):
        train_loader, val_loader = self.create_loaders((X, y))
        for epoch in range(self.epochs):
            epoch_loss, total, correct = 0, 0, 0
            e = 0
            for e, (X, y) in enumerate(train_loader):
                if self.cuda:
                    X, y = X.cuda(), y.cuda()
                output = self.forward(X)
                loss = (
                    self.criterion(output, y) + 1e-3 * self.linear_probe.weight.norm()
                )
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss
                correct += torch.sum(torch.topk(output, dim=1, k=1)[1].squeeze(1) == y)
                total += len(X)
            if (epoch % 10 == 0 or epoch == self.epochs - 1) and verbose:
                print(
                    f"Epoch {epoch}: \t Loss {epoch_loss/(e+1):.3f} \t Acc: {correct/total:.3f}"
                )
            val_acc = self.validation(val_loader)
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_best_model()
        if verbose:
            print(f"Best val acc: {self.best_val_acc:.3f}")

    @torch.no_grad()
    def validation(self, val_loader):
        self.linear_probe.eval()
        correct, total = 0, 0
        for X, y in val_loader:
            if self.cuda:
                X, y = X.cuda(), y.cuda()
            output = self.forward(X)
            correct += torch.sum(
                torch.topk(output, dim=1, k=1)[1].squeeze(1) == y
            ).item()
            total += len(X)
        self.linear_probe.train()
        return correct / total

    def evaluate(self, activations_dataset):
        val_activs = self.create_loaders(activations_dataset, [1.0])[0]
        return self.validation(val_activs)

    def save_best_model(self):
        W = self.linear_probe.state_dict()["weight"].clone().cpu().detach().numpy()
        b = self.linear_probe.state_dict()["bias"].clone().cpu().detach().numpy()
        self.best_model = {"W": W, "b": b}

    def load_best_model(self):
        self.linear_probe.state_dict()["weight"].copy_(
            torch.from_numpy(self.best_model["W"])
        )
        self.linear_probe.state_dict()["bias"].copy_(
            torch.from_numpy(self.best_model["b"])
        )
        if self.cuda:
            self.linear_probe.cuda()


class BaseAnalysis(ABC):
    def __init__(self, rpath, attributes_on_gpu=["backbone"]):
        self.rpath = rpath
        self.cuda = torch.cuda.is_available()
        self.attributes_on_gpu = attributes_on_gpu
        self.empty_variables()
        self.result = {}

    def export(self, name):
        os.makedirs(self.rpath, exist_ok=True)
        torch.save(self.result, os.path.join(self.rpath, name + ".pt"))

    def move_devices_to_cpu(self):
        for attr in self.attributes_on_gpu:
            try:
                a = getattr(self, attr)
                a = a.to("cpu")
                del a
            except AttributeError:
                pass
        torch.cuda.empty_cache()

    def empty_variables(self):
        # cleans the activations, labels, and registered hooks & handles
        self.activs = torch.tensor([])
        self.labels = []
        try:
            for handle in self.handles.values():
                handle.remove()
        except AttributeError:
            pass
        self.handles = {}

    def clean_up(self):
        self.move_devices_to_cpu()
        self.empty_variables()

    @abstractmethod
    def analysis(self):
        pass

    @abstractmethod
    def plot(self, path):
        pass


class RepresentationsAnalysis(BaseAnalysis):
    def __init__(
        self,
        model: Model,
        layers,
        plotter: Callable,
        max_repr_size: int = 8000,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_representation_size = max_repr_size
        self.plotter = plotter
        self.backbone = model
        self.layers_to_analyze = layers
        if self.cuda:
            self.backbone = self.backbone.cuda()
        self.random_indices_for_layers = {}

    def plot(self, name):
        # dependency injection
        # allows us to dynamically change plotting function while creating different analysis objects
        self.plotter(self.result, self.rpath, name)

    def _hook(self, m, i, o, layer_name: str) -> None:
        if layer_name=='lstm':
            out, (h,c) = o
            num_layer = h.shape[0]
            if f"lstm_output" not in self.activs.keys():
                self.activs["lstm_output"]=torch.tensor([])
                output = out[:, -1, :].detach().cpu()
                output = self.subsample_outputs(layer_name, output)
                self.activs["lstm_output"] = torch.cat((self.activs["lstm_output"], output))
            for i in range(num_layer):
                if f"lstm_{i}" not in self.activs.keys():
                    self.activs[f"lstm_{i}"]=torch.tensor([])
                    output = h[i].detach().cpu()
                    output = self.subsample_outputs(layer_name, output)
                    self.activs[f"lstm_{i}"] = torch.cat((self.activs[f"lstm_{i}"], output))
            # import pdb
            # pdb.set_trace()
        else:
            output = o.flatten(start_dim=1).detach().cpu()
            output = self.subsample_outputs(layer_name, output)
            self.activs[layer_name] = torch.cat((self.activs[layer_name], output))

    def subsample_outputs(self, layer_name: str, output: torch.Tensor) -> torch.Tensor:
        """
        Checks whether num features in output is too big (> self.max_representation_size)
        and subsamples them if needed to have max_represntation_size.

        Args:
            layer_name (str): name of layer for extracted representations
            output (torch.Tensor): extracted representations

        Returns:
            output: (torch.Tensor): subsampled (if needed) matrix of representations
        """
        num_features = output.shape[1]
        if num_features > self.max_representation_size:
            self.random_indices_for_layers = self.sample_indices_for_layer(
                layer_name, num_features
            )
            output = output[:, self.random_indices_for_layers[layer_name]]
        return output

    def sample_indices_for_layer(self, layer_name, num_features):
        if layer_name not in self.random_indices_for_layers:
            random_indices = np.random.choice(
                num_features, self.max_representation_size, replace=False
            )
            self.random_indices_for_layers[layer_name] = random_indices
        return self.random_indices_for_layers

    def _insert_hook(self, layer_names:list) -> None:
        for name, layer in self.backbone.named_modules():
            if name in layer_names:
                hook = partial(self._hook, layer_name=name)
                self.handles[name] = layer.register_forward_hook(hook)

    # @torch.no_grad()
    def collect_activations(
        self, loader, layer_names: list, device="cuda:0", eval_hessian=False,
    ):
        self.labels = []
        self._insert_hook(layer_names)
        if 'lstm' not in layer_names:
            self.activs = {layer: torch.tensor([]) for layer in layer_names}
        else:
            self.activs = {}
        
        self.backbone.eval()
        if self.cuda:
            self.backbone = self.backbone.to(device)
        for input, targets, *_ in loader:
            if self.cuda:
                input = input.to(device)
                targets = targets.to(device)
            try:
                self.backbone.forward(input)
            except:
                h = self.backbone.init_hidden(input.shape[0])
                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                h = tuple([each.data for each in h])
                self.backbone.zero_grad()
                output,h = self.backbone(input,h)
            self.labels += [y.item() for y in targets]
        self.labels = torch.tensor(self.labels)
        activs, labels = copy.deepcopy(self.activs), copy.deepcopy(self.labels)
        if eval_hessian:
            self.backbone.train()
            try:
                self.backbone.forward(input)
                criterion = nn.MSELoss()
                loss = criterion(output.squeeze().float(), targets.float())
                loss.backward(create_graph=True)
                hessian_comp = hessian(self.backbone,
                                criterion,
                                dataloader=None,
                                device=device)
            except:
                with torch.backends.cudnn.flags(enabled=False):
                    h = self.backbone.init_hidden(input.shape[0])
                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    h = tuple([each.data for each in h])
                    self.backbone.zero_grad()
                    output,h = self.backbone(input,h)
                    criterion = nn.BCELoss()
                    loss = criterion(output.squeeze().float(), targets.float())
                    loss.backward(create_graph=True)
                    hessian_comp = hessian(self.backbone,
                                criterion,
                                dataloader=None,
                                device=device)

            top_eigenvalues = hessian_comp.eigenvalues()[0]
            trace = hessian_comp.trace()
            top_eigenvalues = top_eigenvalues[0]
            trace = np.mean(trace)
            self.backbone.zero_grad()
            self.backbone.eval()
            import pdb
            pdb.set_trace()
            self.backbone = self.backbone.cpu()
            return activs, labels, top_eigenvalues, trace
        self.backbone = self.backbone.cpu()
        return activs, labels


class Rank(RepresentationsAnalysis):
    def __init__(self, data: Dataset,device="cuda:0", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = torch.utils.data.DataLoader(data, batch_size=256, shuffle=False)
        self.device=device

    def analysis(self, eval_hessian=False):
        # activations, _ = self.collect_activations(self.data, self.layers_to_analyze, device=self.device)
        if eval_hessian:
            activations, _, top_eigenvalues, trace = self.collect_activations(self.data, self.layers_to_analyze, device=self.device, eval_hessian=True,)
            
            # hessian_dataloader = self.data
            # hessian_comp = hessian(self.backbone,
            #                     criterion,
            #                     dataloader=hessian_dataloader,
            #                     cuda=True)

            # top_eigenvalues = hessian_comp.eigenvalues()[0]
            # trace = hessian_comp.trace()
            # top_eigen_dict[domain] = top_eigenvalues[0]
            # trace_avg_dict[domain] = np.mean(trace)
            self.result = {
                "rank": {
                    name: matrix_rank(torch.cov(rep.T)).item()
                    for name, rep in activations.items()
                },
                "top_eigenvalues": top_eigenvalues,
                "trace": trace
            }
        else:
            activations, _ = self.collect_activations(self.data, self.layers_to_analyze, device=self.device, eval_hessian=False,)
            self.result = {
                "rank": {
                    name: matrix_rank(torch.cov(rep.T)).item()
                    for name, rep in activations.items()
                }
            }


class EarlyExit(RepresentationsAnalysis):
    def __init__(self, train_data: Dataset, test_data: Dataset, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_data = train_data
        self.test_data = test_data

    def analysis(self, verbose=False):
        train_loader = DataLoader(self.train_data, batch_size=1024, shuffle=True)
        test_loader = DataLoader(self.test_data, batch_size=1024, shuffle=True)

        for layer_name in self.layers_to_analyze:
            # print(f"Linear probe for layer: {layer_name}.")
            # logger.info(
            #     f"Linear probe for layer: {layer_name}.",
            # )
            X_train, y_train = self.collect_activations(train_loader, [layer_name])
            X_train = X_train[layer_name]
            linear_head = self.train(X_train, y_train, verbose=verbose)

            self.empty_variables()
            X_test, y_test = self.collect_activations(test_loader, [layer_name])
            X_test = X_test[layer_name]
            self.result[layer_name] = linear_head.evaluate((X_test, y_test))
            self.empty_variables()

        return self.result

    def train(self, X, y, verbose=False):
        num_classes = len(set(self.labels))
        num_features = X.shape[1]
        head = LinearProbe(num_features, num_classes)
        head.train(X, y, verbose=verbose)
        return head


if __name__ == "__main__":
    pass