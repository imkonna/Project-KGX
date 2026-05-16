"""Training, evaluation, and full training-loop utilities for GNN experiments.

All functions accept data loaders and device as explicit parameters
so they can be reused across dataset-specific notebooks.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, classification_report,
    r2_score, mean_squared_error,
)

from .featurize import ATOM_FEAT_DIM


# ── Single-epoch training ─────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0
    total_samples = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss = criterion(logits, batch.y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.y.size(0)
        total_samples += batch.y.size(0)
    return total_loss / total_samples


# ── Evaluation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model on a loader.

    Returns
    -------
    tuple : (avg_loss, accuracy, roc_auc, f1, probs_array, labels_array)
    """
    model.eval()
    total_loss = 0
    total_samples = 0
    all_labels, all_probs = [], []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        loss = criterion(logits, batch.y.float())
        total_loss += loss.item() * batch.y.size(0)
        total_samples += batch.y.size(0)

        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(batch.y.cpu().numpy())

    avg_loss = total_loss / total_samples
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds = (all_probs >= 0.5).astype(int)

    acc = accuracy_score(all_labels, preds)
    auc = roc_auc_score(all_labels, all_probs)
    f1 = f1_score(all_labels, preds)

    return avg_loss, acc, auc, f1, all_probs, all_labels


# ── Evaluation (regression) ───────────────────────────────────────────────────
@torch.no_grad()
def evaluate_regression(model, loader, criterion, device):
    """Evaluate regression model. Returns (avg_loss, r2, rmse, preds, labels)."""
    model.eval()
    total_loss = 0
    total_samples = 0
    all_labels, all_preds = [], []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        loss = criterion(out, batch.y.float())
        total_loss += loss.item() * batch.y.size(0)
        total_samples += batch.y.size(0)
        all_preds.extend(out.cpu().numpy())
        all_labels.extend(batch.y.cpu().numpy())

    avg_loss = total_loss / total_samples
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    r2 = r2_score(all_labels, all_preds)
    rmse = float(np.sqrt(mean_squared_error(all_labels, all_preds)))

    return avg_loss, r2, rmse, all_preds, all_labels


# ── Full training loop ────────────────────────────────────────────────────────
def train_model(
    model_class,
    model_name,
    train_loader,
    val_loader,
    test_loader,
    device,
    target_names=("Negative", "Positive"),
    in_dim=ATOM_FEAT_DIM,
    num_epochs=50,
    lr=1e-3,
    weight_decay=1e-4,
    patience=10,
    pos_weight=None,
    task="classification",
    **model_kwargs,
):
    """Full training loop with early stopping and LR scheduling.

    Supports classification (BCEWithLogitsLoss, ROC-AUC) and
    regression (MSELoss, R²/RMSE).  Backward-compatible: existing
    callers that omit *task* get classification by default.
    """
    is_clf = task == "classification"

    print(f"\n{'='*60}")
    print(f"Training: {model_name}  [{task}]")
    print(f"{'='*60}")

    model = model_class(in_dim, **model_kwargs).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if is_clf:
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
        if pos_weight is not None:
            criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight], device=device)
            )
            print(f"Using pos_weight={pos_weight:.2f} for class imbalance")
        else:
            criterion = nn.BCEWithLogitsLoss()
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        criterion = nn.MSELoss()

    best_val_metric = 0.0 if is_clf else float("inf")
    best_state = None
    epochs_no_improve = 0

    if is_clf:
        history = {"train_loss": [], "val_loss": [], "val_auc": [], "val_f1": []}
    else:
        history = {"train_loss": [], "val_loss": [], "val_r2": [], "val_rmse": []}

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        if is_clf:
            val_loss, val_acc, val_auc, val_f1, _, _ = evaluate(
                model, val_loader, criterion, device
            )
            scheduler.step(val_auc)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_auc"].append(val_auc)
            history["val_f1"].append(val_f1)
            improved = val_auc > best_val_metric
            metric_str = f"Val AUC: {val_auc:.4f} | Val F1: {val_f1:.4f}"
        else:
            val_loss, val_r2, val_rmse, _, _ = evaluate_regression(
                model, val_loader, criterion, device
            )
            scheduler.step(val_loss)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_r2"].append(val_r2)
            history["val_rmse"].append(val_rmse)
            improved = val_loss < best_val_metric
            metric_str = f"Val R\u00b2: {val_r2:.4f} | Val RMSE: {val_rmse:.4f}"

        marker = ""
        if improved:
            best_val_metric = val_auc if is_clf else val_loss
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            epochs_no_improve = 0
            marker = "  * Best"
        else:
            epochs_no_improve += 1

        if epoch % 5 == 0 or marker:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:02d}/{num_epochs}  "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"{metric_str} | LR: {current_lr:.1e}{marker}"
            )

        if epochs_no_improve >= patience:
            print(
                f"\nEarly stopping at epoch {epoch} "
                f"(no improvement for {patience} epochs)"
            )
            break

    # Restore best model and evaluate on test set
    model.load_state_dict(best_state)

    if is_clf:
        print(f"\nRestored best model (Val AUC: {best_val_metric:.4f})")
        test_loss, test_acc, test_auc, test_f1, test_probs, test_labels = evaluate(
            model, test_loader, criterion, device
        )
        test_preds = (test_probs >= 0.5).astype(int)
        print(f"\n--- Test Results: {model_name} ---")
        print(f"Accuracy: {test_acc:.4f} | F1: {test_f1:.4f} | ROC AUC: {test_auc:.4f}")
        print(classification_report(test_labels, test_preds, target_names=list(target_names)))
        test_results = {
            "Model": model_name,
            "Accuracy": test_acc,
            "ROC AUC": test_auc,
            "F1 Score": test_f1,
            "probs": test_probs,
            "labels": test_labels,
        }
    else:
        print(f"\nRestored best model (Val Loss: {best_val_metric:.4f})")
        test_loss, test_r2, test_rmse, test_preds, test_labels = evaluate_regression(
            model, test_loader, criterion, device
        )
        print(f"\n--- Test Results: {model_name} ---")
        print(f"R\u00b2: {test_r2:.4f} | RMSE: {test_rmse:.4f}")
        test_results = {
            "Model": model_name,
            "R2": test_r2,
            "RMSE": test_rmse,
            "preds": test_preds,
            "labels": test_labels,
        }

    return model, history, test_results
