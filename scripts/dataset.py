import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioTextDataset(Dataset):
    """
    Dataset for loading audio and text embeddings with corresponding labels.

    Args:
        metadata_csv (str): Path to the metadata CSV file.
        label_col (str): Column name in the CSV containing the labels (default: "label").
        is_regression (bool): If True, treat labels as floats for regression; otherwise, as integers for classification (default: False).
        data_dir (str): Directory containing processed embedding files (default: "data/processed").
        audio_dim (int): Expected dimension of audio embeddings (default: 1024).
        text_dim (int): Expected dimension of text embeddings (default: 768).

    Raises:
        FileNotFoundError: If the metadata CSV file does not exist.
        ValueError: If the label column is not found in the CSV or if embedding dimensions mismatch.
    """
    def __init__(
        self,
        metadata_csv: str,
        label_col: str = "label",
        is_regression: bool = False,
        data_dir: str = "data/processed",
        audio_dim: int = 1024,
        text_dim: int = 768
    ):
        if not os.path.exists(metadata_csv):
            raise FileNotFoundError(f"Metadata CSV file not found: {metadata_csv}")
        
        self.df = pd.read_csv(metadata_csv)
        self.label_col = label_col
        self.is_regression = is_regression
        self.data_dir = data_dir
        self.audio_dim = audio_dim
        self.text_dim = text_dim

        # Validate metadata
        if "participant_id" not in self.df.columns:
            raise ValueError("CSV must contain a 'participant_id' column")
        if label_col not in self.df.columns:
            raise ValueError(f"Label column '{label_col}' not found in CSV")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fetch a single sample from the dataset.

        Args:
            idx (int): Index of the sample to fetch.

        Returns:
            Tuple containing:
                - audio_tensor (torch.Tensor): Audio embedding tensor of shape (T, audio_dim).
                - text_tensor (torch.Tensor): Text embedding tensor of shape (T, text_dim).
                - label_tensor (torch.Tensor): Label tensor (float for regression, long for classification).

        Raises:
            ValueError: If embedding dimensions do not match expected sizes.
        """
        row = self.df.iloc[idx]
        pid = row["participant_id"]
        audio_emb_file = os.path.join(self.data_dir, f"{pid}_audio_emb.npy")
        text_emb_file = os.path.join(self.data_dir, f"{pid}_text_emb.npy")

        # Load or default audio embedding
        if os.path.exists(audio_emb_file):
            audio_emb = np.load(audio_emb_file)
        else:
            logger.warning(f"Audio embedding file missing for participant {pid}, using zeros")
            audio_emb = np.zeros(self.audio_dim)

        # Load or default text embedding
        if os.path.exists(text_emb_file):
            text_emb = np.load(text_emb_file)
        else:
            logger.warning(f"Text embedding file missing for participant {pid}, using zeros")
            text_emb = np.zeros(self.text_dim)

        # Reshape to (T, dim) if necessary
        if len(audio_emb.shape) == 1:
            audio_emb = audio_emb.reshape(1, -1)
        if len(text_emb.shape) == 1:
            text_emb = text_emb.reshape(1, -1)

        # Validate embedding dimensions
        if audio_emb.shape[1] != self.audio_dim:
            raise ValueError(f"Audio embedding dimension mismatch at index {idx}: expected {self.audio_dim}, got {audio_emb.shape[1]}")
        if text_emb.shape[1] != self.text_dim:
            raise ValueError(f"Text embedding dimension mismatch at index {idx}: expected {self.text_dim}, got {text_emb.shape[1]}")

        # Prepare label tensor
        label = row[self.label_col]
        label_dtype = torch.float if self.is_regression else torch.long
        try:
            label_tensor = torch.tensor(label, dtype=label_dtype)
        except (TypeError, ValueError) as e:
            logger.error(f"Invalid label value at index {idx}: {label} (type: {type(label)})")
            raise ValueError(f"Cannot convert label '{label}' to {label_dtype}: {str(e)}")

        # Convert embeddings to tensors
        audio_tensor = torch.tensor(audio_emb, dtype=torch.float)
        text_tensor = torch.tensor(text_emb, dtype=torch.float)

        return audio_tensor, text_tensor, label_tensor

# Example usage
if __name__ == "__main__":
    # Sample test with a dummy CSV
    dummy_csv = "dummy_metadata.csv"
    if not os.path.exists(dummy_csv):
        df = pd.DataFrame({
            "participant_id": ["p1", "p2"],
            "label": [0, 1]
        })
        df.to_csv(dummy_csv, index=False)

    # Instantiate dataset
    dataset = AudioTextDataset(
        metadata_csv=dummy_csv,
        label_col="label",
        is_regression=False,
        data_dir="data/processed",
        audio_dim=1024,
        text_dim=768
    )

    # Test dataset
    print(f"Dataset size: {len(dataset)}")
    for i in range(min(2, len(dataset))):
        audio, text, label = dataset[i]
        print(f"Sample {i}:")
        print(f"  Audio shape: {audio.shape}")
        print(f"  Text shape: {text.shape}")
        print(f"  Label: {label}")
