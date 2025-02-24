#!/usr/bin/env python3
"""
cross_corpus_train.py

Script to perform cross-corpus training and evaluation of the TCF model across interaction types.
Trains on one dataset (e.g., Interview) and tests on others (e.g., TAT, PANSS, DISCOURSE).
"""

import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import logging
from typing import Dict, List, Tuple
from scripts.dataset import AudioTextDataset
from scripts.model_tcf import COLDTCFModelAudioText

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function for TCF model (temporal inputs)."""
    audio_list, text_list, label_list = [], [], []
    for audio, text, label in batch:
        audio_list.append(audio)
        text_list.append(text)
        label_list.append(label)
    return (
        torch.nn.utils.rnn.pad_sequence(audio_list, batch_first=True),
        torch.nn.utils.rnn.pad_sequence(text_list, batch_first=True),
        torch.stack(label_list)
    )

def load_dataset(metadata_csv: str, batch_size: int = 8, audio_dim: int = 1024, text_dim: int = 768) -> DataLoader:
    """Load dataset for a given interaction type."""
    dataset = AudioTextDataset(
        metadata_csv=metadata_csv,
        label_col="label",
        is_regression=False,
        data_dir="data/processed",
        audio_dim=audio_dim,
        text_dim=text_dim
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)

def train_and_evaluate(
    train_loader: DataLoader,
    test_loaders: Dict[str, DataLoader],
    output_dim: int,
    max_epochs: int = 5,
    checkpoint_dir: str = "checkpoints_cross_corpus"
) -> Dict[str, Dict[str, float]]:
    """Train TCF model on one dataset and evaluate on others."""
    # Initialize model
    model = COLDTCFModelAudioText(
        input_dim_audio=1024,
        input_dim_text=768,
        hidden_dim=256,
        output_dim=output_dim,
        is_regression=False,
        lr=1e-3
    )

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="tcf-{epoch:02d}-{val_loss:.2f}",
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
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model, train_loader, train_loader)  # Use train_loader for validation too (simplified)

    # Evaluate on all test datasets
    results = {}
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        for test_name, test_loader in test_loaders.items():
            all_preds, all_labels = [], []
            for audio_x, text_x, labels in test_loader:
                audio_x, text_x, labels = audio_x.to(device), text_x.to(device), labels.to(device)
                preds = model(audio_x, text_x, inference=True)
                all_preds.extend(preds.argmax(dim=-1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            acc = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average="weighted")
            ece = compute_ece(
                torch.tensor(np.array([model(audio_x.to(device), text_x.to(device), inference=True).cpu().numpy() 
                                       for audio_x, text_x, _ in test_loader]).reshape(-1, output_dim)),
                torch.tensor(all_labels),
                n_bins=10
            )
            results[test_name] = {"Accuracy": acc, "F1": f1, "ECE": ece}
            logger.info(f"Test on {test_name} - Accuracy: {acc:.3f}, F1: {f1:.3f}, ECE: {ece:.3f}")

    return results

def main(
    metadata_files: Dict[str, str],
    max_epochs: int = 5,
    batch_size: int = 8,
    output_dir: str = "cross_corpus_results"
):
    """Main function to run cross-corpus training and evaluation."""
    os.makedirs(output_dir, exist_ok=True)
    output_dim = 3  # Assuming 3 classes: low schizotypy, high schizotypy, patient

    all_results = {}
    for train_name, train_csv in metadata_files.items():
        logger.info(f"Training on {train_name}...")
        train_loader = load_dataset(train_csv, batch_size)
        test_loaders = {name: load_dataset(csv, batch_size) for name, csv in metadata_files.items() if name != train_name}
        
        results = train_and_evaluate(
            train_loader,
            test_loaders,
            output_dim,
            max_epochs,
            checkpoint_dir=f"checkpoints_{train_name}"
        )
        all_results[train_name] = results

    # Save results
    df = pd.DataFrame.from_dict(
        {(train, test): all_results[train][test] for train in all_results for test in all_results[train]},
        orient="index"
    )
    df.to_csv(os.path.join(output_dir, "cross_corpus_results.csv"))
    logger.info(f"Saved cross-corpus results to {output_dir}/cross_corpus_results.csv")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Cross-corpus training and evaluation of TCF model.")
    parser.add_argument("--metadata_dir", type=str, required=True, help="Directory with metadata CSVs")
    parser.add_argument("--max_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--output_dir", type=str, default="cross_corpus_results", help="Output directory")
    args = parser.parse_args()

    # Assume metadata files are named by interaction type
    metadata_files = {
        "Interview": os.path.join(args.metadata_dir, "metadata_interview.csv"),
        "TAT": os.path.join(args.metadata_dir, "metadata_tat.csv"),
        "PANSS": os.path.join(args.metadata_dir, "metadata_panss.csv"),
        "DISCOURSE": os.path.join(args.metadata_dir, "metadata_discourse.csv")
    }
    main(metadata_files, args.max_epochs, args.batch_size, args.output_dir)
