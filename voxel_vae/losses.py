"""Plain equal-voxel Cross Entropy plus unnormalized raw KL."""
import torch
import torch.nn.functional as F


def vae_loss(logits, target, mu, logvar, beta=1e-6):
    reconstruction = F.cross_entropy(logits, target.long())
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl = kl.flatten(1).sum(1).mean()
    beta_kl = float(beta) * kl
    total = reconstruction + beta_kl
    return {"total_loss": total, "recon_ce": reconstruction,
            "kl_raw": kl, "beta_kl": beta_kl}


def accuracy_counts(logits, target, air_ids):
    prediction = logits.argmax(dim=1)
    correct = prediction.eq(target)
    non_air = torch.ones_like(target, dtype=torch.bool)
    for air_id in air_ids:
        non_air &= target.ne(int(air_id))
    return {
        "correct": int(correct.sum().item()),
        "voxels": int(target.numel()),
        "non_air_correct": int((correct & non_air).sum().item()),
        "non_air_voxels": int(non_air.sum().item()),
    }
