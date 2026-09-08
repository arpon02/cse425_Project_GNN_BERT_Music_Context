"""
Baseline Models for Comparison (Section 8 of Specification).
Implements:
1. Baseline B1: Random and Majority-Class Tag Predictor
2. Baseline B2: 2D CNN on log-mel spectrogram (no graph, no text)
3. Baseline B3: BERT-only baseline (Task 1)
4. Baseline B4: PCA + MLP on handcrafted audio features (MFCC, Chroma, Spectral Contrast)
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

from src.gnn_model import MelSpectrogramCNN


class RandomBaseline:
    """
    Baseline B1a: Predicts labels according to random uniform or prior probabilities.
    """
    def __init__(self, prior_probs: Optional[np.ndarray] = None, seed: int = 42):
        self.prior_probs = prior_probs
        self.rng = np.random.default_rng(seed)

    def fit(self, y_train: np.ndarray):
        """Fit empirical class frequencies."""
        self.prior_probs = np.mean(y_train, axis=0)

    def predict_proba(self, n_samples: int) -> np.ndarray:
        if self.prior_probs is None:
            return self.rng.uniform(0.0, 1.0, size=(n_samples, 10))
        # Broadcast prior probabilities with slight random jitter
        jitter = self.rng.uniform(-0.05, 0.05, size=(n_samples, len(self.prior_probs)))
        probs = np.clip(self.prior_probs + jitter, 0.01, 0.99)
        return probs

    def predict(self, n_samples: int, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(n_samples) >= threshold).astype(np.float32)


class MajorityClassBaseline:
    """
    Baseline B1b: Always predicts the majority class distribution from training set.
    """
    def __init__(self):
        self.majority_vector = None

    def fit(self, y_train: np.ndarray):
        mean_p = np.mean(y_train, axis=0)
        self.majority_vector = (mean_p >= 0.5).astype(np.float32)

    def predict(self, n_samples: int) -> np.ndarray:
        if self.majority_vector is None:
            return np.zeros((n_samples, 10), dtype=np.float32)
        return np.repeat(self.majority_vector[np.newaxis, :], n_samples, axis=0)


class HandcraftedMLPBaseline(nn.Module):
    """
    Baseline B4: PCA + Multi-Layer Perceptron on handcrafted audio summary features.
    Summarizes audio using global mean and variance of MFCC, chroma, and spectral centroid.
    """
    def __init__(
        self,
        in_dim: int = 32,
        hidden_dim: int = 64,
        num_classes: int = 10,
        dropout: float = 0.2
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
