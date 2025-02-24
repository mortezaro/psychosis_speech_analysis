#!/usr/bin/env python3
"""
evaluate_models.py

Script to evaluate trained models (COLDTCF, EarlyFusion, LateFusion) on a test dataset.
Computes F1, Accuracy, MAE, RMSE, and ECE metrics as applicable.

Usage example:
  python scripts/evaluate_models.py --metadata metadata.csv --model_path model.ckpt --model_type coldtcf --label_column label
"""

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from typing import Dict, List, Tuple
import argparse
import logging
from scripts.model_tcf import COLDTCFModelAudioText
from scripts.fusion_models import EarlyFusionLightning, LateFusionModel, compute_ece
from scripts.dataset import AudioTextDataset

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def collate_fn_tcf(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function for COLDTCF model (temporal inputs)."""
    audio_list, text_list, label_list = [], [], []
    for audio, text, label in batch:
        audio_list.append(audio)
        text_list.append(text)
        label_list.append(label)
    audio_batch = torch.nn.utils.rnn.pad_sequence(audio_list, batch_first=True)  # (B, T, audio_dim)
    text_batch = torch.nn.utils.rnn.pad_sequence(text_list, batch_first=True)    # (B, T, text_dim)
    label_batch = torch.stack(label_list)                                        # (B,)
    return audio_batch, text_batch, label_batch

def collate_fn_fusion(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function for fusion models (static inputs with pooling)."""
    audio_list, text_list, label_list = [], [], []
    for audio, text, label in batch:
        audio_pooled = audio.mean(dim=0)  # (T, audio_dim) -> (audio_dim,)
        text_pooled = text.mean(dim=0)    # (T, text_dim) -> (text_dim,)
        audio_list.append(audio_pooled)
        text_list.append(text_pooled)
        label_list.append(label)
    audio_batch = torch.stack(audio_list)  # (B, audio_dim)
    text_batch = torch.stack(text_list)    # (B, text_dim)
    label_batch = torch.stack(label_list)  # (B,)
    return audio_batch, text_batch, label_batch

def evaluate_model(
    model: pl.LightningModule,
    test_loader: DataLoader,
    is_regression: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, float]:
    """
    Evaluate the model on the test dataset and compute metrics.

    Args:
        model: Trained PyTorch Lightning model.
        test_loader: DataLoader with test data.
        is_regression: If True, compute MAE and RMSE; otherwise, F1, Accuracy, and ECE.
        device: Device to run evaluation on.

    Returns:
        Dict of metric names to values.
    """
    model.eval()
    model.to(device)
    all_preds, all_labels = [], []
    all_logits = []  # For ECE in classification

    with torch.no_grad():
        for batch in test_loader:
            audio_x, text_x, labels = [x.to(device) for x in batch]
            if isinstance(model, COLDTCFModelAudioText):
                preds = model(audio_x, text_x, inference=True)
            else:  # EarlyFusionLightning or LateFusionModel
                preds = model(audio_x, text_x)
            
            if is_regression:
                all_preds.extend(preds.squeeze().cpu().numpy())
            else:
                all_logits.extend(preds.cpu().numpy())  # Logits for ECE
                all_preds.extend(preds.argmax(dim=-1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    metrics = {}
    if is_regression:
        # Regression metrics
        mae = np.mean(np.abs(all_preds - all_labels))
        rmse = np.sqrt(np.mean((all_preds - all_labels) ** 2))
        metrics["MAE"] = mae
        metrics["RMSE"] = rmse
    else:
        # Classification metrics
        f1 = f1_score(all_labels, all_preds, average="weighted")
        accuracy = accuracy_score(all_labels, all_preds)
        ece = compute_ece(torch.tensor(all_logits), torch.tensor(all_labels), n_bins=10, device=device)
        metrics["F1"] = f1
        metrics["Accuracy"] = accuracy
        metrics["ECE"] = ece

    return metrics

def main(
    metadata_csv: str,
    model_path: str,
    model_type: str,
    label_column: str,
    batch_size: int = 8,
    audio_dim: int = 1024,
    text_dim: int = 768,
    hidden_dim: int = 256,
    output_dim: Optional[int] = None,
    is_regression: bool = False
) -> None:
    """
    Main function to evaluate a trained model.

    Args:
        metadata_csv: Path to metadata CSV.
        model_path: Path to trained model checkpoint (.ckpt).
        model_type: "coldtcf", "early", or "late".
        label_column: Column name for labels in CSV.
        batch_size: Batch size for evaluation.
        audio_dim: Audio embedding dimension.
        text_dim: Text embedding dimension.
        hidden_dim: Hidden layer dimension.
        output_dim: Output dimension (inferred if None).
        is_regression: If True, evaluate as regression task.
    """
    # Load dataset
    dataset = AudioTextDataset(
        metadata_csv=metadata_csv,
        label_col=label_column,
        is_regression=is_regression,
        data_dir="data/processed",
        audio_dim=audio_dim,
        text_dim=text_dim
    )

    # Select collate function based on model type
    collate_fn = collate_fn_tcf if model_type.lower() == "coldtcf" else collate_fn_fusion
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )

    # Infer output_dim if not provided
    if output_dim is None:
        output_dim = 1 if is_regression else len(dataset.df[label_column].unique())
        logger.info(f"Inferred output_dim: {output_dim}")

    # Load model
    if model_type.lower() == "coldtcf":
        model = COLDTCFModelAudioText.load_from_checkpoint(
            model_path,
            input_dim_audio=audio_dim,
            input_dim_text=text_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            is_regression=is_regression
        )
    elif model_type.lower() == "early":
        model = EarlyFusionLightning.load_from_checkpoint(
            model_path,
            audio_dim=audio_dim,
            text_dim=text_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )
    elif model_type.lower() == "late":
        model = LateFusionModel.load_from_checkpoint(
            model_path,
            audio_dim=audio_dim,
            text_dim=text_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'coldtcf', 'early', or 'late'.")

    # Evaluate
    metrics = evaluate_model(model, test_loader, is_regression=is_regression)
    logger.info(f"Evaluation results for {model_type} model:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained audio-text models.")
    parser.add_argument("--metadata", type=str, required=True, help="Path to metadata CSV")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint (.ckpt)")
    parser.add_argument("--model_type", type=str, required=True, choices=["coldtcf", "early", "late"], help="Model type")
    parser.add_argument("--label_column", type=str, default="label", help="Column name for labels")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--regression", action="store_true", help="Evaluate as regression task")
    args = parser.parse_args()

    main(
        metadata_csv=args.metadata,
        model_path=args.model_path,
        model_type=args.model_type,
        label_column=args.label_column,
        batch_size=args.batch_size,
        is_regression=args.regression
    )
