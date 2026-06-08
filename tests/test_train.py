"""CPU smoke test for the trainer on tiny synthetic data (no Hydra, no wandb, no network)."""
import numpy as np
import torch
from src.models.deeplob import DeepLOB
from src.train import TrainConfig, class_weights_from_dataset, evaluate, train
from torch.utils.data import TensorDataset


def _tiny_dataset(n=48, seed=0):
    rng = np.random.default_rng(seed)
    X = torch.from_numpy(rng.standard_normal((n, 1, 100, 40)).astype(np.float32))
    y = torch.from_numpy(rng.integers(0, 3, size=n).astype(np.int64))
    return TensorDataset(X, y)


def test_two_epoch_smoke_runs_end_to_end():
    torch.manual_seed(0)
    model = DeepLOB()
    train_ds, val_ds = _tiny_dataset(48, 0), _tiny_dataset(24, 1)
    cfg = TrainConfig(epochs=2, batch_size=16, patience=5, device="cpu")
    result = train(model, train_ds, val_ds, cfg)
    assert result["epochs_ran"] == 2
    assert len(result["history"]) == 2
    assert 0.0 <= result["best_macro_f1"] <= 1.0
    for row in result["history"]:
        assert {"epoch", "train_loss", "val_loss", "macro_f1"} <= row.keys()


def test_class_weights_counter_imbalance():
    # Heavily imbalanced labels -> the rare class gets the largest weight.
    X = torch.randn(30, 1, 100, 40)
    y = torch.tensor([0] * 24 + [1] * 5 + [2] * 1)
    w = class_weights_from_dataset(TensorDataset(X, y))
    assert w.argmax().item() == 2   # class 2 is rarest -> highest weight
    assert w.argmin().item() == 0   # class 0 is most common -> lowest weight


def test_early_stopping_triggers_on_patience():
    # lr=0 keeps the model (TLOB uses LayerNorm, no BatchNorm running-stat drift) unchanged each epoch, so
    # val macro-F1 never improves after epoch 0 and early stopping must fire well before the epoch cap.
    from src.models.tlob import TLOB

    model = TLOB(dim=24, depth=1, heads=4)
    train_ds, val_ds = _tiny_dataset(32, 2), _tiny_dataset(16, 3)
    cfg = TrainConfig(epochs=10, batch_size=16, patience=1, lr=0.0, device="cpu", use_class_weights=False)
    result = train(model, train_ds, val_ds, cfg)
    assert result["epochs_ran"] < 10  # stopped before the cap
    assert result["best_epoch"] == 0


def test_evaluate_returns_macro_f1_and_loss():
    model = DeepLOB()
    ds = _tiny_dataset(16, 4)
    from torch.utils.data import DataLoader

    loader = DataLoader(ds, batch_size=8)
    out = evaluate(model, loader, torch.nn.CrossEntropyLoss(), torch.device("cpu"))
    assert 0.0 <= out["macro_f1"] <= 1.0
    assert out["loss"] >= 0.0
