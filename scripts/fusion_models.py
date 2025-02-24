import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from typing import Optional, Tuple

def compute_ece(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 10,
    device: str = "cpu"
) -> float:
    """
    Compute Expected Calibration Error (ECE) for classification.
    
    Args:
        logits: Tensor of shape (B, num_classes) with model logits.
        labels: Tensor of shape (B,) with ground truth labels.
        n_bins: Number of bins for confidence calibration (default: 10).
        device: Device to perform computation on (default: "cpu").
    
    Returns:
        Float representing the ECE value.
    
    Raises:
        ValueError: If logits and labels have incompatible shapes or if n_bins < 1.
    """
    if logits.shape[0] != labels.shape[0]:
        raise ValueError(f"Batch size mismatch: logits ({logits.shape[0]}) vs labels ({labels.shape[0]})")
    if n_bins < 1:
        raise ValueError(f"n_bins must be positive, got {n_bins}")

    probs = F.softmax(logits, dim=-1)
    conf, pred = torch.max(probs, dim=-1)
    conf, pred, labels = conf.to(device), pred.to(device), labels.to(device)

    bins = torch.linspace(0, 1, n_bins + 1, device=device)
    ece = torch.tensor(0.0, device=device)

    for i in range(n_bins):
        bin_lower, bin_upper = bins[i], bins[i + 1]
        idxs = (conf >= bin_lower) & (conf < bin_upper)
        n_in_bin = idxs.sum()

        if n_in_bin > 0:
            acc_in_bin = (pred[idxs] == labels[idxs]).float().mean()
            avg_conf_in_bin = conf[idxs].mean()
            bin_frac = n_in_bin / len(conf)
            ece += torch.abs(acc_in_bin - avg_conf_in_bin) * bin_frac

    return ece.item()

class EarlyFusionModel(nn.Module):
    """Simple early fusion model concatenating audio and text features."""
    def __init__(self, audio_dim: int, text_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(audio_dim + text_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, audio_x: torch.Tensor, text_x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for early fusion.
        
        Args:
            audio_x: Tensor of shape (B, audio_dim)
            text_x: Tensor of shape (B, text_dim)
        
        Returns:
            Tensor of shape (B, output_dim)
        """
        if audio_x.shape[0] != text_x.shape[0]:
            raise ValueError("Batch sizes of audio and text inputs must match")
        fused = torch.cat([audio_x, text_x], dim=-1)
        x = self.relu(self.fc1(fused))
        return self.fc2(x)

class LateFusionAudio(nn.Module):
    """Audio branch for late fusion model."""
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        return self.fc2(x)

class LateFusionText(nn.Module):
    """Text branch for late fusion model."""
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        return self.fc2(x)

class LateFusionModel(pl.LightningModule):
    """Late fusion model combining audio and text branches with a trainable weight."""
    def __init__(
        self,
        audio_dim: int,
        text_dim: int,
        hidden_dim: int,
        output_dim: int,
        lr: float = 1e-3,
        alpha_init: float = 0.5
    ):
        super().__init__()
        self.save_hyperparameters()
        self.audio_module = LateFusionAudio(audio_dim, hidden_dim, output_dim)
        self.text_module = LateFusionText(text_dim, hidden_dim, output_dim)
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.lr = lr

    def forward(self, audio_x: torch.Tensor, text_x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for late fusion.
        
        Args:
            audio_x: Tensor of shape (B, audio_dim)
            text_x: Tensor of shape (B, text_dim)
        
        Returns:
            Tensor of shape (B, output_dim)
        """
        if audio_x.shape[0] != text_x.shape[0]:
            raise ValueError("Batch sizes of audio and text inputs must match")
        audio_out = self.audio_module(audio_x)
        text_out = self.text_module(text_x)
        alpha_clamped = torch.sigmoid(self.alpha)  # Ensure alpha stays in [0, 1]
        return alpha_clamped * audio_out + (1 - alpha_clamped) * text_out

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step with cross-entropy loss."""
        audio_x, text_x, labels = batch
        logits = self(audio_x, text_x)
        loss = F.cross_entropy(logits, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int):
        """Validation step with loss and ECE computation."""
        audio_x, text_x, labels = batch
        logits = self(audio_x, text_x)
        loss = F.cross_entropy(logits, labels)
        ece = compute_ece(logits, labels, n_bins=10, device=self.device)
        self.log_dict({"val_loss": loss, "val_ece": ece}, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure Adam optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.lr)

class EarlyFusionLightning(pl.LightningModule):
    """Early fusion model integrated with PyTorch Lightning."""
    def __init__(
        self,
        audio_dim: int,
        text_dim: int,
        hidden_dim: int,
        output_dim: int,
        lr: float = 1e-3
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = EarlyFusionModel(audio_dim, text_dim, hidden_dim, output_dim)
        self.lr = lr

    def forward(self, audio_x: torch.Tensor, text_x: torch.Tensor) -> torch.Tensor:
        return self.model(audio_x, text_x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step with cross-entropy loss."""
        audio_x, text_x, labels = batch
        logits = self(audio_x, text_x)
        loss = F.cross_entropy(logits, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int):
        """Validation step with loss and ECE computation."""
        audio_x, text_x, labels = batch
        logits = self(audio_x, text_x)
        loss = F.cross_entropy(logits, labels)
        ece = compute_ece(logits, labels, n_bins=10, device=self.device)
        self.log_dict({"val_loss": loss, "val_ece": ece}, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure Adam optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.lr)

# Example usage
if __name__ == "__main__":
    # Dummy data for testing
    batch_size, audio_dim, text_dim, hidden_dim, output_dim = 32, 512, 768, 256, 10
    audio_x = torch.randn(batch_size, audio_dim)
    text_x = torch.randn(batch_size, text_dim)
    labels = torch.randint(0, output_dim, (batch_size,))

    # Test Early Fusion
    early_model = EarlyFusionLightning(audio_dim, text_dim, hidden_dim, output_dim)
    early_logits = early_model(audio_x, text_x)
    print(f"Early Fusion Output Shape: {early_logits.shape}")

    # Test Late Fusion
    late_model = LateFusionModel(audio_dim, text_dim, hidden_dim, output_dim)
    late_logits = late_model(audio_x, text_x)
    print(f"Late Fusion Output Shape: {late_logits.shape}")

    # Test ECE
    ece_value = compute_ece(late_logits, labels)
    print(f"ECE: {ece_value:.4f}")

    # Simulate training with PyTorch Lightning
    trainer = pl.Trainer(max_epochs=5, accelerator="auto", devices=1)
    dummy_batch = (audio_x, text_x, labels)
    trainer.fit(late_model, train_dataloaders=[dummy_batch], val_dataloaders=[dummy_batch])
