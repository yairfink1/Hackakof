import torch
import torch.nn as nn


class ModelArchitecture(nn.Module):
    """
    Lightweight CNN model architecture for local training from scratch.
    """

    def __init__(self, num_classes: int = 20):
        super().__init__()

        # Input shape: [batch_size, 3, 224, 224]
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> [batch_size, 16, 112, 112]

            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> [batch_size, 32, 56, 56]

            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> [batch_size, 64, 28, 28]

            # Block 4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # -> [batch_size, 128, 4, 4]
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Batch of images of shape [batch_size, 3, 224, 224]

        Returns:
            Logits of shape [batch_size, 20]
        """
        x = self.features(x)
        logits = self.classifier(x)
        return logits