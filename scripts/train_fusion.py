#!/usr/bin/env python3
"""
train_fusion.py

Script to train fusion models (EarlyFusionLightning or LateFusionModel) from fusion_models.py.
Saves the best model checkpoint based on validation loss.

Usage example:
  python scripts/train_fusion.py --metadata metadata.csv --model_type early --label_column label
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from typing import Optional, Tuple, List
from scripts.fusion_models import EarlyFusionLightning, LateFusionModel
from scripts.dataset import AudioTextDataset
from sklearn.model_selection import train_test_split
import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function to pad sequences and pool over time dimension for static fusion models.

    Args:
        batch: List of (audio_tensor, text_tensor, label_tensor) tuples.

    Returns:
        Tuple of batched tensors: (audio_batch, text_batch, label_batch).
    """
    audio_list, text_list, label_list = [], [], []
    for audio, text, label in batch:
        # Pool over time dimension (T) to get (dim,)
        audio_pooled = audio.mean(dim=0)  # (T, audio_dim) -> (audio_dim,)
        text_pooled = text.mean(dim=0)    # (T, text_dim) -> (text_dim,)
        audio_list.append(audio_pooled)
        text_list.append(text_pooled)
        label_list.append(label)

    audio_batch = torch.stack(audio_list)  # (B, audio_dim)
    text_batch = torch.stack(text_list)    # (B, text_dim)
    label_batch = torch.stack(label_list)  # (B,)
    return audio_batch, text_batch, label_batch

def run_training(
    metadata_csv: str,
    model_type: str = "early",
    label_column: str = "label",
    batch_size: int = 8,
    max_epochs: int = 5,
    audio_dim: int = 1024,
    text_dim: int = 768,
    hidden_dim: int = 256,
    output_dim: Optional[int] = None,
    lr: float = 1e-3
) -> None:
    """
    Run training for the specified fusion model and save the best checkpoint.

    Args:
        metadata_csv (str): Path to metadata CSV.
        model_type (str): "early" or "late" to select fusion model.
        label_column (str): Column name for labels in CSV.
        batch_size (int): Batch size for training.
        max_epochs (int): Number of training epochs.
        audio_dim (int): Audio embedding dimension.
        text_dim (int): Text embedding dimension.
        hidden_dim (int): Hidden layer dimension.
        output_dim (Optional[int]): Output dimension (inferred if None).
        lr (float): Learning rate.
    """
    # Load dataset
    dataset = AudioTextDataset(
        metadata_csv=metadata_csv,
        label_col=label_column,
        is_regression=False,  # Fusion models assume classification
        data_dir="data/processed",
        audio_dim=audio_dim,
        text_dim=text_dim
    )

    # Train/validation split
    indices = range(len(dataset))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(train_idx),
        collate_fn=collate_fn,
        num_workers=4
    )
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(val_idx),
        collate_fn=collate_fn,
        num_workers=4
    )

    # Determine output dimension if not provided
    if output_dim is None:
        output_dim = len(dataset.df[label_column].unique())
        logger.info(f"Inferred output_dim: {output_dim} classes")

    # Initialize model
    if model_type.lower() == "early":
        model = EarlyFusionLightning(
            audio_dim=audio_dim,
            text_dim=text_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            lr=lr
        )
    elif model_type.lower() == "late":
        model = LateFusionModel(
            audio_dim=audio_dim,
            text_dim=text_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            lr=lr,
            alpha_init=0.5
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'early' or 'late'.")

    # Define checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename=f"{model_type}-{{epoch:02d}}-{{val_loss:.2f}}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        verbose=True
    )

    # Train
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=10,
        enable_progress_bar=True,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model, train_loader, val_loader)
    logger.info(f"Training completed for {model_type} model. Best checkpoint saved at: {checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EarlyFusionLightning or LateFusionModel.")
    parser.add_argument("--metadata", type=str, default="metadata.csv", help="Path to metadata CSV")
    parser.add_argument("--model_type", type=str, default="early", choices=["early", "late"], help="Model type: 'early' or 'late'")
    parser.add_argument("--label_column", type=str, default="label", help="Column name for labels")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--max_epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    run_training(
        metadata_csv=args.metadata,
        model_type=args.model_type,
        label_column=args.label_column,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        lr=args.lr
    )

