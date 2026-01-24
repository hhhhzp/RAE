"""
Dataset class for loading cached latents.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple


class CachedLatentDataset(Dataset):
    """
    Dataset for loading pre-cached VAE latents.

    Args:
        cached_path: Directory containing cached .npz files
        use_flip: Whether to randomly use flipped latents (default: True)
    """

    def __init__(self, cached_path: str, use_flip: bool = True):
        self.cached_path = Path(cached_path)
        self.use_flip = use_flip

        # Find all cached latent files
        self.samples = sorted(self.cached_path.glob("latent_*.npz"))

        if len(self.samples) == 0:
            raise ValueError(f"No cached latents found in {cached_path}")

        print(f"Found {len(self.samples)} cached latent files in {cached_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Args:
            index: Index

        Returns:
            tuple: (latent, label)
        """
        path = self.samples[index]

        # Load cached data
        data = np.load(path)

        # Randomly choose between normal and flipped latent
        if self.use_flip and torch.rand(1).item() < 0.5:
            latent = data['latent_flip']
        else:
            latent = data['latent']

        label = int(data['label'])

        # Convert to tensor
        latent = torch.from_numpy(latent)

        return latent, label
