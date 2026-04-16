"""
Task 3.2: Training Pipeline
Task 3.3: Evaluation Metrics

Training script for the failure prediction network.

Performs:
  - Dataset loading
  - Forward pass
  - Loss computation (Binary Cross Entropy for failure, MSE for error)
  - Backpropagation
  - Evaluation: accuracy, precision, recall, F1, ROC AUC

Usage:
    python -m src.models.train --config configs/config.yaml
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import sys
import argparse
import json
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.failure_predictor import build_model
from src.data.dataset_builder import load_dataset, build_dataset, save_dataset
from src.utils.config_loader import load_config


class EarlyStopping:
    def __init__(self, patience: int = 10):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            return True  # improved
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    threshold: float = 0.5) -> dict:
    """Compute classification metrics (Task 3.3).

    Returns: accuracy, precision, recall, F1, ROC AUC
    """
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score)

    y_binary = (y_pred >= threshold).astype(int)

    metrics = {
        'accuracy': accuracy_score(y_true, y_binary),
        'precision': precision_score(y_true, y_binary, zero_division=0),
        'recall': recall_score(y_true, y_binary, zero_division=0),
        'f1': f1_score(y_true, y_binary, zero_division=0),
    }

    try:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred)
    except ValueError:
        metrics['roc_auc'] = 0.0

    return metrics


def train_epoch(model, dataloader, optimizer, device, bce_weight=1.0, mse_weight=0.5):
    """Train one epoch."""
    model.train()
    total_loss = 0
    bce_fn = nn.BCELoss()
    mse_fn = nn.MSELoss()

    for X_batch, y_fail_batch, y_err_batch in dataloader:
        X_batch = X_batch.to(device)
        y_fail_batch = y_fail_batch.to(device)
        y_err_batch = y_err_batch.to(device)

        optimizer.zero_grad()
        fail_pred, err_pred = model(X_batch)

        loss_bce = bce_fn(fail_pred.squeeze(), y_fail_batch)
        loss_mse = mse_fn(err_pred.squeeze(), y_err_batch)
        loss = bce_weight * loss_bce + mse_weight * loss_mse

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)

    return total_loss / len(dataloader.dataset)


def evaluate(model, dataloader, device):
    """Evaluate model on a dataset."""
    model.eval()
    bce_fn = nn.BCELoss()
    mse_fn = nn.MSELoss()

    all_fail_preds = []
    all_fail_targets = []
    all_err_preds = []
    all_err_targets = []
    total_loss = 0

    with torch.no_grad():
        for X_batch, y_fail_batch, y_err_batch in dataloader:
            X_batch = X_batch.to(device)
            y_fail_batch = y_fail_batch.to(device)
            y_err_batch = y_err_batch.to(device)

            fail_pred, err_pred = model(X_batch)

            loss_bce = bce_fn(fail_pred.squeeze(), y_fail_batch)
            loss_mse = mse_fn(err_pred.squeeze(), y_err_batch)
            loss = loss_bce + 0.5 * loss_mse
            total_loss += loss.item() * X_batch.size(0)

            all_fail_preds.append(fail_pred.squeeze().cpu().numpy())
            all_fail_targets.append(y_fail_batch.cpu().numpy())
            all_err_preds.append(err_pred.squeeze().cpu().numpy())
            all_err_targets.append(y_err_batch.cpu().numpy())

    all_fail_preds = np.concatenate(all_fail_preds)
    all_fail_targets = np.concatenate(all_fail_targets)
    all_err_preds = np.concatenate(all_err_preds)
    all_err_targets = np.concatenate(all_err_targets)

    avg_loss = total_loss / len(dataloader.dataset)
    cls_metrics = compute_metrics(all_fail_targets, all_fail_preds)
    mae = np.mean(np.abs(all_err_targets - all_err_preds))

    return avg_loss, cls_metrics, mae


def make_dataloader(split_data: dict, batch_size: int, shuffle: bool = True):
    """Create DataLoader from dataset split."""
    X = torch.FloatTensor(split_data['X'])
    y_failure = torch.FloatTensor(split_data['y_failure'])
    y_error = torch.FloatTensor(split_data['y_error'])
    ds = TensorDataset(X, y_failure, y_error)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train(config: dict):
    """Main training loop."""
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load or build dataset
    dataset_path = os.path.join(config['paths']['results_dir'], 'train_dataset.pkl')
    if os.path.exists(dataset_path):
        print(f"Loading dataset from {dataset_path}")
        dataset = load_dataset(dataset_path)
    else:
        print("Building dataset...")
        dataset = build_dataset(config)
        save_dataset(dataset, dataset_path)

    # DataLoaders
    batch_size = config['training']['batch_size']
    train_loader = make_dataloader(dataset['train'], batch_size, shuffle=True)
    val_loader = make_dataloader(dataset['val'], batch_size, shuffle=False)
    test_loader = make_dataloader(dataset['test'], batch_size, shuffle=False)

    print(f"\nTrain: {len(dataset['train']['X'])} samples")
    print(f"Val:   {len(dataset['val']['X'])} samples")
    print(f"Test:  {len(dataset['test']['X'])} samples")

    # Model
    model = build_model(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {config['model']['type']} ({n_params:,} parameters)")

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    early_stopping = EarlyStopping(patience=config['training']['early_stopping_patience'])

    # Training loop
    best_model_state = None
    history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'val_auc': []}

    print("\nTraining...")
    for epoch in range(config['training']['epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_metrics, val_mae = evaluate(model, val_loader, device)
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_metrics['f1'])
        history['val_auc'].append(val_metrics['roc_auc'])

        improved = early_stopping.step(val_loss)
        if improved:
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | F1: {val_metrics['f1']:.3f} | "
                  f"AUC: {val_metrics['roc_auc']:.3f} | MAE: {val_mae:.4f}"
                  f"{' *' if improved else ''}")

        if early_stopping.should_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model and evaluate on test set
    if best_model_state:
        model.load_state_dict(best_model_state)

    test_loss, test_metrics, test_mae = evaluate(model, test_loader, device)

    print(f"\n{'='*60}")
    print(f"Test Results ({config['model']['type'].upper()}):")
    print(f"  Loss:      {test_loss:.4f}")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1 Score:  {test_metrics['f1']:.4f}")
    print(f"  ROC AUC:   {test_metrics['roc_auc']:.4f}")
    print(f"  Error MAE: {test_mae:.4f}")
    print(f"{'='*60}")

    # Save model and results
    model_dir = config['paths']['model_save_dir']
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{config['model']['type']}_failure_predictor.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'test_metrics': test_metrics,
        'test_mae': test_mae,
        'history': history,
    }, model_path)
    print(f"\nModel saved to {model_path}")

    # Save results JSON
    results = {
        'model_type': config['model']['type'],
        'test_metrics': test_metrics,
        'test_mae': float(test_mae),
        'test_loss': float(test_loss),
        'n_params': n_params,
        'epochs_trained': len(history['train_loss']),
    }
    results_path = os.path.join(config['paths']['results_dir'],
                                f"{config['model']['type']}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    return model, results


def main():
    parser = argparse.ArgumentParser(description='Train Failure Prediction Network')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--model', type=str, choices=['mlp', 'lstm', 'transformer'],
                        default=None, help='Override model type from config')
    args = parser.parse_args()

    config = load_config(args.config)
    if args.model:
        config['model']['type'] = args.model

    train(config)


if __name__ == '__main__':
    main()
