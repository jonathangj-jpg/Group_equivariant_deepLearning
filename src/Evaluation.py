import csv
from pathlib import Path

import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Preprocessing import load_qm9
from E3Model import Network

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


@torch.no_grad()
def evaluate(model, loader, y_mean, y_std, target):
    ...
    
def train(
    model,
    train_loader,
    val_loader,
    y_mean,
    y_std,
    target,
    epochs=20,
    lr=1e-3,
    weight_decay=1e-8,
):
    ...
    

def save_results(history, test_mse, out_dir="results"):
    ...
    

def plot_history(history, out_dir="results"):
    ...