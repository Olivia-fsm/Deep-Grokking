import sys
import os

import torch
from torch.utils.data import random_split

from datasets.continual_scenarios import prepare_tasks
from tunnel_methods import EarlyExit, Rank
from utils import (
    gin_config_to_dict,
    create_backbone,
    add_multiheads,
    get_list_of_layers,
    load_model_weights,
    ModelConfig,
    seed_everything,
)


Dataset = torch.utils.data.Dataset
Model = torch.nn.Module
Tensor = torch.Tensor


def subset(data_data: Dataset, ratio: float) -> Dataset:
    for key, val in data_data.items():
        split_size = (int(len(val) * ratio), len(val) - int(len(val) * ratio))
        data_data[key] = random_split(val, split_size)[0]
    return data_data


def get_subsets_of_datasets(
    *data_dicts: list, ratios
) -> list:
    """
    ratio can be a list (with different value for each dataset) if its not a list, just extand it to a list
    """
    subset_dicts = []
    if isinstance(ratios, float):
        ratios = [ratios] * len(data_dicts)
    for ratio, data_dict in zip(ratios, data_dicts):  # type: ignore
        subset_dicts.append(subset(data_dict, ratio))
    return subset_dicts


def create_ood_data(args):
    if args["dataset"] == "cifar10" or args["dataset"] == "cinic10":
        train_datasets_ood, _, test_datasets_ood, _ = prepare_tasks(
            data_dir=args["data_dir"],
            dataset="cifar100",
            tasks_sizes=[10, 90],
            train_data_ratio=args["train_data_ratio"],
        )
    elif args["dataset"] == "cifar100":
        train_datasets_ood, _, test_datasets_ood, _ = prepare_tasks(
            data_dir=args["data_dir"],
            dataset="cifar10",
            tasks_sizes=[10],
            train_data_ratio=args["train_data_ratio"],
        )
    else:
        raise ValueError("Unknown dataset choose from: (cifar10|cinic10|cifar100)")
    return train_datasets_ood, test_datasets_ood


def early_exits(
    rpath: str,
    model: Model,
    train_data: Dataset,
    test_data: Dataset,
    layers: list,
    name: str,
) -> None:
    early_exit = EarlyExit(
        model=model,
        train_data=train_data,  # type: ignore
        test_data=test_data,
        layers=layers,
        rpath=rpath,
        plotter=lambda x: x + 1,
    )

    early_exit.analysis()
    early_exit.export(name=name)
    # early_exit.plot(name=name)
    early_exit.clean_up()


def rank_analysis(
    rpath: str, model: Model, dataset: Dataset, layers: list, name: str
) -> None:
    rank = Rank(
        model=model, data=dataset, layers=layers, rpath=rpath, plotter=lambda x: x + 1
    )
    rank.analysis()
    rank.export(name)
    # rank.plot(name)
    rank.clean_up()


def main(ckpt_dname, ckpt_fname, analysis_to_perform=None):
    """
    analysis_to_perform=None runs all the analysis otherwise run only selected
    """
    if analysis_to_perform is None:
        from config import tasks

        analysis_to_perform = tasks

    # load args from the experiment and create config
    args = gin_config_to_dict(ckpt_dname, "config")
    args["num_tasks"] = len(args["tasks_sizes"])
    model_config = ModelConfig.from_dict(args)

    # create a directory for results
    rpath = os.path.join(ckpt_dname, "analysis")
    os.makedirs(rpath, exist_ok=True)

    # prepare datasets with split defined by the seed
    seed_everything(args["seed"])
    train_datasets, _, test_datasets, _ = prepare_tasks(
        data_dir=args["data_dir"],
        dataset=args["dataset"],
        tasks_sizes=args["tasks_sizes"],
        train_data_ratio=args["train_data_ratio"],
    )

    # create model & load weights from the checkpoint
    model, _ = create_backbone(model_config, args["dataset"])
    model = add_multiheads(model, args["tasks_sizes"])
    try:
        model = load_model_weights(model, ckpt_dname, ckpt_fname, only_features=False)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Could not load weights for checkpoint {os.path.join(ckpt_dname, ckpt_fname)}"
        )

    layers_for_analysis = get_list_of_layers(model, exclude_list=["multihead"])

    train_data, test_data = get_subsets_of_datasets(
        train_datasets, test_datasets, ratios=0.2  # type: ignore
    )

    if "rank" in analysis_to_perform:
        rank_analysis(
            rpath=rpath,
            model=model,
            dataset=test_data[0],
            layers=layers_for_analysis,
            name="rank",
        )

    if "early_exits_ID" in analysis_to_perform:
        early_exits(
            rpath=rpath,
            model=model,
            train_data=train_data[0],
            test_data=test_data[0],
            layers=layers_for_analysis,
            name="early_exits_ID",
        )

    if "early_exits_OOD" in analysis_to_perform:
        ood_train_data, ood_test_data = create_ood_data(args)
        early_exits(
            rpath=rpath,
            model=model,
            train_data=ood_train_data[0],
            test_data=ood_test_data[0],
            layers=layers_for_analysis,
            name="early_exits_OOD",
        )


if __name__ == "__main__":
    analysis_to_perform = None
    if len(sys.argv) == 2:
        ckpt_path = sys.argv[1]
        ckpt_name = [
            i
            for i in os.listdir(ckpt_path)
            if os.path.isdir(os.path.join(ckpt_path, i))
        ]

    elif len(sys.argv) == 3:
        ckpt_path, ckpt_name = sys.argv[1], sys.argv[2]
        ckpt_name = [ckpt_name]
    elif len(sys.argv) > 3:
        ckpt_path, ckpt_name, analysis_to_perform = (
            sys.argv[1],
            sys.argv[2],
            sys.argv[3:],
        )
        ckpt_name = [ckpt_name]
    else:
        raise ValueError(
            "Please provide at least one argument (path to root directory of the results)"
        )

    for ckpt_fname in ckpt_name:
        main(ckpt_path, ckpt_fname, analysis_to_perform)
