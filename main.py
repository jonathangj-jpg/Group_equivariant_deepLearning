import csv
import os

import torch

from src.Preprocessing import load_qm9
from src.E3Model import Network
from src.Evaluation import train, device

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]
SUBSETS = [5000, 25000]
LMAXES = [0, 1]
DATA_SEED = 1747
TARGET = 4
BATCH_SIZE = 32
EPOCHS = 100
OUT_DIR = "results"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/test.csv", "w", newline="") as test_file:
        test_writer = csv.writer(test_file)
        test_writer.writerow(["lmax", "subset", "seed", "test_mae_eV"])
        for subset in SUBSETS:
            for lmax in LMAXES:
                with open(f"{OUT_DIR}/l{lmax}_n{subset}.csv", "w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(["seed", "epoch", "train_mse", "train_mae_eV", "val_mae_eV"])
                    for seed in SEEDS:
                        train_loader, val_loader, test_loader, y_mean, y_std = load_qm9(
                            target=TARGET, subset=subset, batch_size=BATCH_SIZE, seed=DATA_SEED
                        )
                        torch.manual_seed(seed)
                        model = Network(lmax=lmax)

                        print(f"=== l={lmax} n={subset} seed={seed} ===")

                        _, history, _, test_mae = train(
                            model, train_loader, val_loader, test_loader,
                            y_mean, y_std, TARGET, epochs=EPOCHS,
                        )

                        if device.type == "cuda":
                            torch.cuda.empty_cache() # This seemed to help gpu memory management

                        for epoch, train_mse, train_mae, val_mae in history:
                            writer.writerow([seed, epoch, train_mse, train_mae, val_mae])

                        file.flush()
                        test_writer.writerow([lmax, subset, seed, test_mae])
                        test_file.flush()
                        print(f"l={lmax} n={subset} seed={seed} | test MAE {test_mae:.4f} eV")


if __name__ == "__main__":
    main()
