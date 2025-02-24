import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import Optional, Tuple
from model_tcf import COLDTCFModelAudioText
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioTextDataset(Dataset):
    """Dataset for loading audio and text embeddings with labels."""
    def __init__(
        self,
        metadata_csv: str,
        data_dir: str = "data/processed",
        is_regression: bool = True,
        label_column: str = "label",
        audio_dim: int = 1024,
        text_dim: int = 768
    ):
        if not os.path.exists(metadata_csv):
            raise FileNotFoundError(f"Metadata file {metadata_csv} not found.")
        self.df = pd.read_csv(metadata_csv)
        self.data_dir = data_dir
        self.is_regression = is_regression
        self.label_column = label_column
        self.audio_dim = audio_dim
        self.text_dim = text_dim

        if self.label_column not in self.df.columns:
            raise ValueError(f"Label column '{self.label_column}' not found in metadata.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        pid = row["participant_id"]
        audio_emb_file = os.path.join(self.data_dir, f"{pid}_audio_emb.npy")
        text_emb_file = os.path.join(self.data_dir, f"{pid}_text_emb.npy")

        if os.path.exists(audio_emb_file):
            audio_emb = np.load(audio_emb_file)
        else:
            logger.warning(f"Audio embedding not found for {pid}, using zeros.")
            audio_emb = np.zeros(self.audio_dim)

        if os.path.exists(text_emb_file):
            text_emb = np.load(text_emb_file)
        else:
            logger.warning(f"Text embedding not found for {pid}, using zeros.")
            text_emb = np.zeros(self.text_dim)

        if len(audio_emb.shape) == 1:
            audio_emb = audio_emb.reshape(1, -1)
        if len(text_emb.shape) == 1:
            text_emb = text_emb.reshape(1, -1)

        if audio_emb.shape[1] != self.audio_dim or text_emb.shape[1] != self.text_dim:
            raise ValueError(f"Embedding dimension mismatch at index {idx}: audio {audio_emb.shape}, text {text_emb.shape}")

        label = row[self.label_column]
        label_dtype = torch.float if self.is_regression else torch.long

        return (
            torch.tensor(audio_emb, dtype=torch.float),
            torch.tensor(text_emb, dtype=torch.float),
            torch.tensor(label, dtype=label_dtype)
        )

def collate_fn(batch: list) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function to pad variable-length sequences."""
    audio_list, text_list, label_list = [], [], []
    for audio, text, label in batch:
        audio_list.append(audio)
        text_list.append(text)
        label_list.append(label)

    audio_tensor = torch.nn.utils.rnn.pad_sequence(audio_list, batch_first=True)
    text_tensor = torch.nn.utils.rnn.pad_sequence(text_list, batch_first=True)
    labels_tensor = torch.stack(label_list)

    return audio_tensor, text_tensor, labels_tensor

def run_training(
    metadata_csv: str,
    is_regression: bool = True,
    label_column: str = "label",
    batch_size: int = 8,
    max_epochs: int = 5,
    audio_dim: int = 1024,
    text_dim: int = 768,
    hidden_dim: int = 256,
    output_dim: Optional[int] = None,
    lambda_co: float = 1.0,
    lambda_reg: float = 1.0,
    lr: float = 1e-3
):
    """Run training for the audio-text model."""
    dataset = AudioTextDataset(
        metadata_csv=metadata_csv,
        is_regression=is_regression,
        label_column=label_column,
        audio_dim=audio_dim,
        text_dim=text_dim
    )
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)

    if output_dim is None:
        output_dim = 1 if is_regression else len(dataset.df[label_column].unique())

    model = COLDTCFModelAudioText(
        input_dim_audio=audio_dim,
        input_dim_text=text_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        is_regression=is_regression,
        lambda_co=lambda_co,
        lambda_reg=lambda_reg,
        lr=lr
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=10
    )
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train the COLDTCF audio-text model.")
    parser.add_argument("--metadata", type=str, default="metadata.csv", help="Path to metadata CSV")
    parser.add_argument("--regression", action="store_true", help="Run as regression task")
    parser.add_argument("--label_column", type=str, default="label", help="Column name for labels")
    args = parser.parse_args()

    run_training(
        metadata_csv=args.metadata,
        is_regression=args.regression,
        label_column=args.label_column
    )
