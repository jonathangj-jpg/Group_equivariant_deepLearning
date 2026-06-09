import torch
from e3nn import o3
import e3nn
import torch_geometric
from sklearn.model_selection import GroupShuffleSplit

from figure_dataset import generate_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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

    dataset = torch_geometric.datasets.QM9(root=root)

    # Reproducible shuffle, then optional subset for small-data experiments.
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

    # Standardization statistics computed on the train split only (no leakage).
    y_train = torch.cat([d.y[:, target] for d in train_set])
    y_mean, y_std = y_train.mean(), y_train.std()

    train_loader = torch.geometric.loader.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.geometric.loader.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.geometric.loader.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, y_mean, y_std
