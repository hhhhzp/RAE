import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class FeatureAlignmentLoss(nn.Module):
    """
    Feature alignment loss between encoder features and predicted features.
    Uses DINOv2 with registers as the feature encoder.
    """

    def __init__(
        self, encoder_name="facebook/dinov2-with-registers-base", device="cuda"
    ):
        super().__init__()
        from transformers import Dinov2WithRegistersModel

        # Initialize encoder
        self.encoder = Dinov2WithRegistersModel.from_pretrained(
            encoder_name, local_files_only=True
        )
        self.encoder = self.encoder.to(device)
        self.encoder.eval()
        self.encoder.requires_grad_(False)

        # ImageNet normalization
        self.normalize = Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)

        self.unused_token_num = 5  # 1 CLS + 4 register tokens

    def extract_features(self, images):
        """
        Extract features from images using the encoder.

        Args:
            images: [B, C, H, W] tensor of images in [0, 1] range

        Returns:
            features: [B, M, C] tensor of features
        """
        # Normalize images
        normalized_images = self.normalize(images)

        # Extract features
        with torch.no_grad():
            encoder_output = self.encoder(normalized_images)
            image_features = encoder_output.last_hidden_state[
                :, self.unused_token_num :
            ]
            # Reshape from [B, N, C] to [B, H, W, C] assuming square feature map
            B, N, C = image_features.shape
            H = W = int(N**0.5)
            image_features = image_features.reshape(B, H, W, C).permute(0, 3, 1, 2)
            # Pixel unshuffle: [B, C, H, W] -> [B, C*4, H/2, W/2]
            image_features = F.pixel_unshuffle(image_features, downscale_factor=2)
            image_features = image_features.flatten(2, 3)
        return image_features.permute(0, 2, 1)

    def forward(self, images, pred_features):
        """
        Compute feature alignment loss.

        Args:
            images: [B, C, H, W] tensor of images in [0, 1] range
            pred_features: [B, M, C] tensor of predicted features

        Returns:
            loss: scalar tensor
        """
        # Extract features from images
        image_features = self.extract_features(images)

        # Normalize and compute cosine similarity loss
        image_features_norm = F.normalize(image_features, dim=-1)
        pred_features_norm = F.normalize(pred_features, dim=-1)
        cos_sim = (image_features_norm * pred_features_norm).sum(dim=-1)  # [B, M]
        loss = -cos_sim.mean()

        return loss
