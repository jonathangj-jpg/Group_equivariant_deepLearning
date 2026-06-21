import torch
from skopt import gp_minimize
from skopt.space import Real, Categorical

from src.Preprocessing import load_qm9
from src.E3Model import Network
from src.Evaluation import train, device

TARGET = 4
SUBSET = 10000
LMAX = 0
EPOCHS = 100
DATA_SEED = 915
MODEL_SEED = 0
N_CALLS = 30
N_INITIAL = 10

space = [
    Real(1e-4, 5e-3, prior="log-uniform", name="lr"),
    Real(1e-6, 1e-2, prior="log-uniform", name="weight_decay"),
    Real(0.0, 0.5, name="dropout"),
    Real(3, 5, name="r"),
    Categorical([8, 16, 32], name="batch_size"),
]


def objective(x):
    lr, weight_decay, dropout, r, batch_size = x

    train_loader, val_loader, test_loader, y_mean, y_std = load_qm9(
        target=TARGET, subset=SUBSET, batch_size=int(batch_size), seed=DATA_SEED
    )

    torch.manual_seed(MODEL_SEED)
    model = Network(lmax=LMAX, dropout=float(dropout), r=float(r))

    _, _, best_val, _ = train(
        model, train_loader, val_loader, test_loader,
        y_mean, y_std, TARGET, epochs=EPOCHS,
        lr=float(lr), weight_decay=float(weight_decay),
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return best_val


def progress(result):
    call = len(result.func_vals)
    print(f"call {call}/{N_CALLS} | best {result.fun:.4f} eV", flush=True)


def main():
    result = gp_minimize(
        objective, space,
        n_calls=N_CALLS,
        n_initial_points=N_INITIAL,
        acq_func="EI",
        random_state=0,
        callback=progress,
    )

    names = [d.name for d in space]

    print(f"l={LMAX}")
    print(f"Best validation MAE: {result.fun:.4f} eV")
    print("Best hyperparameters:")
    for name, val in zip(names, result.x):
        print(f"  {name}: {val}")


if __name__ == "__main__":
    main()
