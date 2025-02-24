import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

def distance_function_regression(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute squared difference between predictions and targets for regression."""
    return (preds.squeeze() - targets.float()) ** 2

def distance_function_classification(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute negative log likelihood for classification."""
    log_probs = F.log_softmax(logits, dim=-1)
    return -log_probs[torch.arange(len(targets)), targets.long()]

def kl_div_bi_direction(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute bidirectional KL divergence between two distributions with numerical stability."""
    p, q = p.clamp(min=eps), q.clamp(min=eps)  # Avoid log(0)
    kl_pq = (p * (torch.log(p) - torch.log(q))).sum()
    kl_qp = (q * (torch.log(q) - torch.log(p))).sum()
    return kl_pq + kl_qp

def compute_calib_ordinality_loss(
    audio_preds: torch.Tensor,
    text_preds: torch.Tensor,
    labels: torch.Tensor,
    audio_var_norm: torch.Tensor,
    text_var_norm: torch.Tensor,
    is_regression: bool = True
) -> torch.Tensor:
    """Compute calibration ordinality loss for audio and text predictions."""
    if is_regression:
        dist_a = distance_function_regression(audio_preds, labels)
        dist_t = distance_function_regression(text_preds, labels)
    else:
        dist_a = distance_function_classification(audio_preds, labels)
        dist_t = distance_function_classification(text_preds, labels)

    dist_a_soft = F.softmax(-dist_a, dim=0)
    var_a_soft  = F.softmax(-audio_var_norm, dim=0)
    dist_t_soft = F.softmax(-dist_t, dim=0)
    var_t_soft  = F.softmax(-text_var_norm, dim=0)

    kl_a = kl_div_bi_direction(dist_a_soft, var_a_soft)
    kl_t = kl_div_bi_direction(dist_t_soft, var_t_soft)

    dist_at = torch.cat([dist_a, dist_t], dim=0)
    var_at  = torch.cat([audio_var_norm, text_var_norm], dim=0)
    dist_at_soft = F.softmax(-dist_at, dim=0)
    var_at_soft  = F.softmax(-var_at, dim=0)

    kl_at = kl_div_bi_direction(dist_at_soft, var_at_soft)
    return kl_a + kl_t + kl_at

class GRUDistributionBlock(nn.Module):
    """GRU-based block to model distributions with mean and log-variance outputs."""
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.mean_head = nn.Linear(hidden_dim, hidden_dim)
        self.logvar_head = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out, _ = self.gru(x)
        mu = self.mean_head(out)
        logvar = self.logvar_head(out)
        return mu, logvar

class COLDTCFModelAudioText(pl.LightningModule):
    """Multimodal model for audio and text with calibration and ordinality loss."""
    def __init__(
        self,
        input_dim_audio: int = 1024,
        input_dim_text: int = 768,
        hidden_dim: int = 256,
        output_dim: int = 1,
        is_regression: bool = True,
        lambda_co: float = 1.0,
        lambda_reg: float = 1.0,
        lr: float = 1e-3,
        max_var: float = 1e4
    ):
        super().__init__()
        self.save_hyperparameters()
        self.is_regression = is_regression
        self.output_dim = output_dim
        self.lr = lr
        self.max_var = max_var

        self.audio_dist = GRUDistributionBlock(input_dim_audio, hidden_dim)
        self.text_dist = GRUDistributionBlock(input_dim_text, hidden_dim)
        self.output_head = nn.Linear(hidden_dim, output_dim)
        self.audio_head = nn.Linear(hidden_dim, output_dim)
        self.text_head = nn.Linear(hidden_dim, output_dim)

        self.lambda_co = lambda_co
        self.lambda_reg = lambda_reg

    def variance_regularizer(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence to regularize variance towards a standard normal."""
        var = torch.exp(logvar)
        kl = 0.5 * torch.sum(mu**2 + var - logvar - 1.0, dim=1)
        return kl.mean()

    def fuse_timewise(
        self,
        mu_a: torch.Tensor,
        var_a: torch.Tensor,
        mu_t: torch.Tensor,
        var_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fuse audio and text representations based on normalized variances."""
        var_a_norm = var_a.norm(dim=2, keepdim=True)
        var_t_norm = var_t.norm(dim=2, keepdim=True)
        denom = var_a_norm + var_t_norm + 1e-9
        w_a = var_t_norm / denom
        w_t = var_a_norm / denom
        fused = w_a * mu_a + w_t * mu_t
        return fused, var_a_norm.squeeze(-1), var_t_norm.squeeze(-1)

    def forward(self, audio_x: torch.Tensor, text_x: torch.Tensor, inference: bool = False):
        """Forward pass for training or inference."""
        mu_a, logvar_a = self.audio_dist(audio_x)
        mu_t, logvar_t = self.text_dist(text_x)

        var_a = torch.exp(logvar_a).clamp(1e-6, self.max_var)
        var_t = torch.exp(logvar_t).clamp(1e-6, self.max_var)

        fused, var_a_norm, var_t_norm = self.fuse_timewise(mu_a, var_a, mu_t, var_t)
        fused_mean = fused.mean(dim=1)
        fused_preds = self.output_head(fused_mean)

        if inference:
            return fused_preds

        audio_mean = mu_a.mean(dim=1)
        text_mean = mu_t.mean(dim=1)
        audio_preds = self.audio_head(audio_mean)
        text_preds = self.text_head(text_mean)
        audio_var_norm = var_a_norm.mean(dim=1)
        text_var_norm = var_t_norm.mean(dim=1)

        return fused_preds, {
            "audio_preds": audio_preds,
            "text_preds": text_preds,
            "mu_a": mu_a, "var_a": var_a, "logvar_a": logvar_a,
            "mu_t": mu_t, "var_t": var_t, "logvar_t": logvar_t,
            "audio_var_norm": audio_var_norm,
            "text_var_norm": text_var_norm
        }

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Training step with main loss, calibration loss, and regularization."""
        audio_x, text_x, labels = batch
        fused_preds, extras = self(audio_x, text_x)

        if self.is_regression:
            main_loss = F.mse_loss(fused_preds.squeeze(), labels.float())
        else:
            main_loss = F.cross_entropy(fused_preds, labels.long())

        co_loss = compute_calib_ordinality_loss(
            extras["audio_preds"], extras["text_preds"],
            labels, extras["audio_var_norm"], extras["text_var_norm"],
            is_regression=self.is_regression
        )
        co_loss_weighted = self.lambda_co * co_loss

        reg_a = self.variance_regularizer(extras["mu_a"], extras["logvar_a"])
        reg_t = self.variance_regularizer(extras["mu_t"], extras["logvar_t"])
        reg_loss = self.lambda_reg * (reg_a + reg_t)

        loss = main_loss + co_loss_weighted + reg_loss

        self.log_dict({
            "train_main_loss": main_loss,
            "train_co_loss": co_loss,
            "train_reg_loss": reg_loss,
            "train_total_loss": loss
        }, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Validation step with loss computation."""
        audio_x, text_x, labels = batch
        fused_preds = self(audio_x, text_x, inference=True)
        val_loss = F.mse_loss(fused_preds.squeeze(), labels.float()) if self.is_regression else F.cross_entropy(fused_preds, labels.long())
        self.log("val_loss", val_loss, on_epoch=True)
        return val_loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure Adam optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.lr)
