import copy

import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


@torch.no_grad()
def evaluate(model, loader, y_mean, y_std, target):
    model.eval()
    total_abs_error = 0.0
    total_samples = 0

    for data in loader:
        data = data.to(device)
        pred = model(data).squeeze(-1) * y_std + y_mean
        target_y = data.y[:, target]
        total_abs_error += (pred - target_y).abs().sum().item()
        total_samples += target_y.numel()

    return total_abs_error / total_samples
    
def train(
    model,
    train_loader,
    val_loader,
    test_loader,
    y_mean,
    y_std,
    target,
    epochs=100,
    lr=1e-3,
    weight_decay=1e-4,
):
    model = model.to(device)
    y_mean = y_mean.to(device)
    y_std = y_std.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )
    history = []
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_abs_error = 0.0
        total_samples = 0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            pred = model(data).squeeze(-1)
            target_y = (data.y[:, target] - y_mean) / y_std
            loss = F.mse_loss(pred, target_y)

            loss.backward()
            optimizer.step()

            batch_size = target_y.numel()
            total_loss += loss.item() * batch_size
            total_abs_error += (pred.detach() - target_y).abs().sum().item()
            total_samples += batch_size

        train_mse = total_loss / total_samples
        train_mae = (total_abs_error / total_samples) * y_std.item()
        val_mae = evaluate(model, val_loader, y_mean, y_std, target)
        history.append((epoch, train_mse, train_mae, val_mae))
        print(f"epoch {epoch:3d} | train MSE {train_mse:.4f} | train MAE {train_mae:.4f} eV | val MAE {val_mae:.4f} eV")

        if val_mae < best_val:
            best_val = val_mae
            best_state = copy.deepcopy(model.state_dict())

        scheduler.step()

    model.load_state_dict(best_state)
    test_mae = evaluate(model, test_loader, y_mean, y_std, target)

    return model, history, best_val, test_mae