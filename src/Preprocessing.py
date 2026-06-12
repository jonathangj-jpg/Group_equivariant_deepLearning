import logging

import torch
from torch_cluster import radius_graph
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import QM9

from e3nn import o3
from e3nn.nn import FullyConnectedNet, Gate
from e3nn.o3 import FullyConnectedTensorProduct
from e3nn.math import soft_one_hot_linspace
from e3nn.util.test import assert_equivariant

def load_figure(n_points=100, n_samples=10, n_augmentations=10):
    ...
    
def load_qm9(
    root="./qm9_data",
    target=4,
    batch_size=64,
    train_frac=0.8,
    val_frac=0.1,
    subset=None,
    seed=42,
):

    dataset = QM9(root=root)

    # Reproducible shuffle
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(dataset), generator=generator)
    if subset is not None:
        perm = perm[:subset]
    dataset = dataset[perm]

    n = len(dataset)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)

    train_set = dataset[:n_train]
    val_set = dataset[n_train : n_train + n_val]
    test_set = dataset[n_train + n_val :]

    # Standardization computed on the train split only (no dataleakage)
    y_train = torch.cat([d.y[:, target] for d in train_set])
    y_mean, y_std = y_train.mean(), y_train.std()

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, y_mean, y_std

def build_qm9_graph_inputs():
    ...

load_qm9()