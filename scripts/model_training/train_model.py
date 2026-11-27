"""Обучение нейронной сети (MLP) для кредитного скоринга.

Пайплайн:
  1. Загрузка train/validation из data/processed.
  2. Стандартизация признаков (StandardScaler, сохраняется для инференса).
  3. Обучение MLP на PyTorch с early stopping по val ROC-AUC.
  4. Оценка на test: ROC-AUC, PR-AUC, accuracy, F1, KS.
  5. Сохранение: models/trained/model.pt, scaler.pkl, metrics.json.

Запуск:
    python scripts/model_training/train_model.py --config config/params.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_generation import FEATURE_NAMES, TARGET  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_model")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class CreditMLP(nn.Module):
    """MLP-классификатор: [in] -> Linear+BN+ReLU+Dropout -> ... -> Linear(1) -> sigmoid."""

    def __init__(self, n_features: int, hidden_dims: list[int],
                 dropout: float, use_batch_norm: bool, activation: str = "relu"):
        super().__init__()
        act = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}[activation]()
        layers: list[nn.Module] = []
        d_in = n_features
        for d_out in hidden_dims:
            layers.append(nn.Linear(d_in, d_out))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(d_out))
            layers.append(act)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d_in = d_out
        layers.append(nn.Linear(d_in, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(-1)


def load_split(path: str, scaler: StandardScaler | None = None):
    df = pd.read_csv(path)
    X = df[FEATURE_NAMES].values.astype(np.float32)
    y = df[TARGET].values.astype(np.float32)
    if scaler is None:
        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X).astype(np.float32)
        return Xs, y, scaler
    return scaler.transform(X).astype(np.float32), y, scaler


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), 4096):
        out.append(model(torch.from_numpy(X[i:i + 4096]).to(device)).cpu().numpy())
    return np.concatenate(out)


def evaluate(y_true: np.ndarray, proba: np.ndarray) -> dict:
    pred = (proba >= 0.5).astype(int)
    # Индекс Колмогорова–Смирнова — стандартная метрика скоринговых карт:
    # максимум расхождения CDF положительного и отрицательного классов
    p_pos = np.sort(proba[y_true == 1])
    p_neg = np.sort(proba[y_true == 0])
    grid = np.unique(np.concatenate([p_pos, p_neg]))
    cdf_pos = np.searchsorted(p_pos, grid, side="right") / max(len(p_pos), 1)
    cdf_neg = np.searchsorted(p_neg, grid, side="right") / max(len(p_neg), 1)
    ks = float(np.max(np.abs(cdf_pos - cdf_neg)))
    return {
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 5),
        "pr_auc": round(float(average_precision_score(y_true, proba)), 5),
        "accuracy": round(float(accuracy_score(y_true, pred)), 5),
        "f1": round(float(f1_score(y_true, pred)), 5),
        "ks_statistic": round(ks, 5),
        "n_samples": int(len(y_true)),
    }


def main(cfg: dict) -> None:
    tcfg, mcfg = cfg["training"], cfg["model"]
    set_seed(tcfg["random_state"])
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log.info("Устройство обучения: %s", device)

    train_dir = cfg["paths"]["processed_data"]
    X_tr, y_tr, scaler = load_split(os.path.join(train_dir, "train.csv"))
    X_va, y_va, _ = load_split(os.path.join(train_dir, "validation.csv"), scaler)
    X_te, y_te, _ = load_split(os.path.join(train_dir, "test.csv"), scaler)
    log.info("train=%d val=%d test=%d, признаков=%d", len(X_tr), len(X_va), len(X_te), X_tr.shape[1])

    model = CreditMLP(
        n_features=X_tr.shape[1],
        hidden_dims=mcfg["hidden_dims"],
        dropout=mcfg["dropout"],
        use_batch_norm=mcfg["use_batch_norm"],
        activation=mcfg["activation"],
    ).to(device)
    log.info("Архитектура:\n%s", model)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=tcfg["batch_size"], shuffle=True, drop_last=False,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg["learning_rate"],
                                 weight_decay=tcfg["weight_decay"])
    criterion = nn.BCELoss()

    best_auc, best_state, patience = -1.0, None, 0
    history = []
    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += float(loss) * len(xb)
        val_proba = predict(model, X_va, device)
        val_auc = roc_auc_score(y_va, val_proba)
        history.append({"epoch": epoch, "train_loss": round(total / len(X_tr), 5),
                        "val_roc_auc": round(val_auc, 5)})
        log.info("epoch %02d | loss=%.4f | val ROC-AUC=%.4f", epoch, total / len(X_tr), val_auc)
        if val_auc > best_auc:
            best_auc, patience = val_auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= tcfg["early_stopping_patience"]:
                log.info("Early stopping на эпохе %d (best val AUC=%.4f)", epoch, best_auc)
                break

    model.load_state_dict(best_state)
    test_metrics = evaluate(y_te, predict(model, X_te, device))
    val_metrics = evaluate(y_va, predict(model, X_va, device))
    log.info("TEST: %s", test_metrics)

    os.makedirs(cfg["paths"]["trained_models"], exist_ok=True)
    out_dir = cfg["paths"]["trained_models"]
    torch.save({
        "state_dict": model.state_dict(),
        "arch": {"n_features": X_tr.shape[1], "hidden_dims": mcfg["hidden_dims"],
                 "dropout": mcfg["dropout"], "use_batch_norm": mcfg["use_batch_norm"],
                 "activation": mcfg["activation"]},
        "feature_names": FEATURE_NAMES,
    }, os.path.join(out_dir, "model.pt"))
    import joblib
    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))

    framework_str = "torch/" + torch.__version__
    summary = {
        "validation": val_metrics, "test": test_metrics,
        "best_val_roc_auc": round(float(best_auc), 5),
        "training_history": history,
        "model_type": "pytorch_mlp", "framework": framework_str,
        "feature_names": FEATURE_NAMES,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log.info("Модель и метрики сохранены в %s/", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/params.yaml")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as fh:
        main(yaml.safe_load(fh))
