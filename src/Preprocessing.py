import torch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import QM9
    
def load_qm9(
    root="./qm9_data",
    target=4,
    batch_size=64,
    val_size=5000,
    test_size=5000,
    subset=None,
    seed=42,
):

    dataset = QM9(root=root)

    # Reproducible shuffle
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(dataset), generator=generator)

    # Fixed-size val/test held out first, so every subset size is validated and
    # tested on the exact same molecules; `subset` only controls the train size
    val_set = dataset[perm[:val_size]]
    test_set = dataset[perm[val_size : val_size + test_size]]
    train_perm = perm[val_size + test_size :]
    if subset is not None:
        train_perm = train_perm[:subset]
    train_set = dataset[train_perm]

    # Standardization computed on the train split only (no dataleakage)
    y_train = torch.cat([d.y[:, target] for d in train_set])
    y_mean, y_std = y_train.mean(), y_train.std()

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, y_mean, y_std