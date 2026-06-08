"""Training loop for LOB direction models (DeepLOB / TLOB).

Config-driven (a ``TrainConfig`` dataclass; a Hydra entrypoint wraps it), with Adam, class-weighted
cross-entropy for the imbalanced stationary class, macro-F1 early stopping, checkpointing, and optional
Weights & Biases logging. The core ``train`` function takes plain objects so it is unit-testable on CPU with
no Hydra and no network.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

__all__ = ["TrainConfig", "build_model", "class_weights_from_dataset", "evaluate", "train"]


@dataclass
class TrainConfig:
    """Trainer configuration."""

    epochs: int = 50
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 8
    device: str = "cpu"
    num_workers: int = 0
    use_class_weights: bool = True
    grad_clip: float | None = 1.0
    ckpt_path: str | None = None
    seed: int = 42
    wandb_enabled: bool = False
    wandb_project: str = "deep-lob"


def build_model(name: str, **kwargs) -> nn.Module:
    """Instantiate a model by name ('deeplob' or 'tlob')."""
    key = name.lower()
    if key == "deeplob":
        from src.models.deeplob import DeepLOB

        return DeepLOB(**kwargs)
    if key == "tlob":
        from src.models.tlob import TLOB

        return TLOB(**kwargs)
    raise ValueError(f"unknown model {name!r}; choose 'deeplob' or 'tlob'")


def class_weights_from_dataset(dataset: Dataset, n_classes: int = 3) -> torch.Tensor:
    """Inverse-frequency class weights (normalized) to counter label imbalance."""
    if hasattr(dataset, "class_counts"):
        counts = np.asarray(dataset.class_counts(), dtype=float)
    else:
        counts = np.zeros(n_classes, dtype=float)
        for i in range(len(dataset)):  # type: ignore[arg-type]
            counts[int(dataset[i][1])] += 1.0
    counts = np.clip(counts, 1.0, None)
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict:
    """Return macro-F1 and mean loss over a loader."""
    from sklearn.metrics import f1_score

    model.eval()
    preds: list[int] = []
    targets: list[int] = []
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += float(criterion(logits, y)) * y.size(0)
        n += y.size(0)
        preds.extend(logits.argmax(1).cpu().tolist())
        targets.extend(y.cpu().tolist())
    macro_f1 = float(f1_score(targets, preds, labels=[0, 1, 2], average="macro", zero_division=0))
    return {"macro_f1": macro_f1, "loss": total_loss / max(n, 1)}


def train(model: nn.Module, train_ds: Dataset, val_ds: Dataset, cfg: TrainConfig) -> dict:
    """Train with class-weighted CE, macro-F1 early stopping, and best-checkpoint restore.

    Returns a dict with ``best_macro_f1``, ``best_epoch``, ``epochs_ran``, and per-epoch ``history``.
    """
    _seed_everything(cfg.seed)
    device = torch.device(cfg.device)
    model.to(device)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=False
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    weight = class_weights_from_dataset(train_ds).to(device) if cfg.use_class_weights else None
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    run = None
    if cfg.wandb_enabled:
        import wandb

        run = wandb.init(project=cfg.wandb_project, config=vars(cfg))

    best_f1, best_epoch, best_state = -1.0, -1, None
    no_improve, history = 0, []

    for epoch in range(cfg.epochs):
        model.train()
        running, seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            if cfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            running += float(loss) * y.size(0)
            seen += y.size(0)
        train_loss = running / max(seen, 1)

        val = evaluate(model, val_loader, criterion, device)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val["loss"],
               "macro_f1": val["macro_f1"]}
        history.append(row)
        if run is not None:
            run.log(row)

        if val["macro_f1"] > best_f1:
            best_f1, best_epoch = val["macro_f1"], epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
            if cfg.ckpt_path:
                torch.save(best_state, cfg.ckpt_path)
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)  # restore best
    if run is not None:
        run.finish()

    return {
        "best_macro_f1": best_f1,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "history": history,
    }


def main() -> None:
    """Hydra entrypoint (lazy import so the module is usable/testable without Hydra)."""
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../configs", config_name="train")
    def _run(dcfg: DictConfig) -> None:
        from src.data.fi2010 import FI2010Dataset, load_fi2010_file

        train_arr = load_fi2010_file(dcfg.data.train_path)
        test_arr = load_fi2010_file(dcfg.data.test_path)
        train_ds = FI2010Dataset(train_arr, horizon=dcfg.data.horizon)
        val_ds = FI2010Dataset(test_arr, horizon=dcfg.data.horizon)
        model = build_model(dcfg.model.name, **dict(dcfg.model.get("kwargs", {})))
        cfg = TrainConfig(**dict(dcfg.train))
        result = train(model, train_ds, val_ds, cfg)
        print(f"best macro-F1 {result['best_macro_f1']:.4f} at epoch {result['best_epoch']}")

    _run()


if __name__ == "__main__":
    main()
