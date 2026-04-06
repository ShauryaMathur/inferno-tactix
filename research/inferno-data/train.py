"""
Wildfire Prediction CNN-LSTM Model Trainer

This script trains a Convolutional Neural Network (CNN) followed by a Bidirectional Long Short-Term 
Memory (BiLSTM) network to predict the occurrence of wildfires based on time-series 
environmental data.

Usage:
  python train.py --csv_path Wildfire_Dataset.csv --epochs 100

Author:  Shreyas Bellary Manjunath <> Shaurya Mathur
Date:    2025-05-01
"""

from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("trainer")

#Data Preparation ─────────────────────────────────────────────────
def load_and_preprocess_data(csv_path: str, seq_len: int, features: list[str]) -> tuple:
    log.info(f"Loading data from '{csv_path}'...")
    try:
        df = pd.read_csv(csv_path, parse_dates=["datetime"])
    except FileNotFoundError:
        log.error(f"Error: The file '{csv_path}' was not found.")
        raise

    #Removes rows with the placeholder fill value
    fill_value = 32767.0
    mask = ~(df == fill_value).any(axis=1)
    df = df.loc[mask].reset_index(drop=True)

    df = df.sort_values(["latitude", "longitude", "datetime"]).reset_index(drop=True)
    df["seq_id"] = np.arange(len(df)) // seq_len
    
    seq_counts = df['seq_id'].value_counts()
    complete_seq_ids = seq_counts[seq_counts == seq_len].index
    df = df[df['seq_id'].isin(complete_seq_ids)]

    groups = list(df.groupby("seq_id"))
    if not groups:
        log.error("No complete sequences of length %d found. Check SEQ_LEN or data integrity.", seq_len)
        raise ValueError("Could not create sequences from the data.")

    log.info(f"Created {len(groups)} sequences of length {seq_len}.")

    seqs = np.stack([g[features].values for _, g in groups], axis=0)
    labels = np.array([int((g["Wildfire"] == "Yes").any()) for _, g in groups])

    X_temp, X_test, y_temp, y_test = train_test_split(
        seqs, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    if np.unique(y_temp).size > 1:
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
        )
    else: 
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.2, random_state=42
        )


    log.info(
        f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}"
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


#Dataset & DataLoader ──────────────────────────────────────────────
class WildfireDataset(Dataset):

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


#Model Architecture ───────────────────────────────────────────────
class CNN_LSTM_Wildfire(nn.Module):
    def __init__(self, input_features: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_features, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=2),
        )
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,   
        )
        self.fc = nn.Linear(hidden_size * 2, 1) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        _, (hn, _) = self.lstm(x)                            
        out = torch.cat((hn[-2], hn[-1]), dim=1)             
        return self.fc(out).squeeze(1)


#Train and Evaluate Functions ──────────────────────────────────────
def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0, 0, 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)

        optimizer.zero_grad()
        logits = model(Xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * Xb.size(0)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == yb).sum().item()
        total += yb.size(0)

    return total_loss / total, correct / total


def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits = model(Xb)
            loss = criterion(logits, yb)

            total_loss += loss.item() * Xb.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == yb).sum().item()
            total += yb.size(0)

    return total_loss / total, correct / total


#Main Pipeline ──────────────────────────────────────────
def main(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data(
        args.csv_path, args.seq_len, args.features
    )

    train_loader = DataLoader(
        WildfireDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        WildfireDataset(X_val, y_val), batch_size=args.batch_size
    )
    test_loader = DataLoader(
        WildfireDataset(X_test, y_test), batch_size=args.batch_size
    )

    model = CNN_LSTM_Wildfire(
        input_features=len(args.features),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    log.info("Model Architecture:\n%s", model)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = -1

    log.info("Starting model training...")
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        log.info(
            f"Epoch {epoch+1:03}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_epoch = epoch
            torch.save(model.state_dict(), args.save_path)
            log.info(f"Validation loss improved. Saving model to '{args.save_path}'")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            log.info(
                f"Early stopping triggered at epoch {epoch+1} due to no improvement in validation loss."
            )
            break

    log.info("Loading best model for final evaluation on the test set.")
    try:
        model.load_state_dict(torch.load(args.save_path))
    except FileNotFoundError:
        log.error(f"Could not load the model from '{args.save_path}'. Please check the path.")
        return

    final_test_loss, final_test_acc = eval_epoch(model, test_loader, criterion, device)
    log.info(f"Best model was saved from epoch {best_epoch+1}.")
    log.info(
        f"Final Test Results -> Loss: {final_test_loss:.4f}, Accuracy: {final_test_acc:.4f}"
    )


#CLI Parser ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a CNN-LSTM model for wildfire prediction."
    )

    parser.add_argument(
        "--csv_path",
        type=str,
        default="Wildfire_Dataset.csv",
        help="Path to the wildfire dataset CSV file.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="best_wildfire_model.pth",
        help="Path to save the best model weights.",
    )

    parser.add_argument(
        "--seq_len", type=int, default=75, help="Length of input sequences."
    )
    parser.add_argument(
        "--hidden_size", type=int, default=128, help="LSTM hidden size."
    )
    parser.add_argument("--num_layers", type=int, default=3, help="Number of LSTM layers.")
    parser.add_argument("--dropout", type=float, default=0.25, help="Dropout rate.")
    
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Patience for early stopping.",
    )
    
    parser.add_argument(
        '--features',
        nargs='+',
        default=['pr','rmax','rmin','sph','srad','tmmn','tmmx','vs','bi','fm100','fm1000','erc','pet','vpd'],
        help='List of feature columns to use from the CSV.'
    )

    cli_args = parser.parse_args()
    main(cli_args)