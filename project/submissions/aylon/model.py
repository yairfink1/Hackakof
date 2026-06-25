import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool=False):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.block(x)

class ModelArchitecture(nn.Module):
    """
    ResNet-9 with proper ImageNet stem for 224x224 inputs.

    The stem uses a 7x7 conv with stride 2 + MaxPool to aggressively
    downsample from 224x224 → 56x56 before the heavy conv blocks.
    This dramatically reduces VRAM usage and speeds up training
    WITHOUT sacrificing accuracy (early layers only learn trivial
    edge detectors that don't need full spatial resolution).
    """
    def __init__(self, num_classes: int = 20):
        super().__init__()
        
        # Stem: aggressively downsample 224 → 56
        # This is what standard ImageNet ResNets do.
        # 224 → 112 (stride-2 conv) → 56 (maxpool)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 56x56
        )
        
        # Layer 1: 56x56 → 28x28
        self.layer1 = ConvBlock(64, 128, pool=True)
        self.res1 = nn.Sequential(
            ConvBlock(128, 128),
            ConvBlock(128, 128)
        )
        
        # Layer 2: 28x28 → 14x14
        self.layer2 = ConvBlock(128, 256, pool=True)
        
        # Layer 3: 14x14 → 7x7
        self.layer3 = ConvBlock(256, 512, pool=True)
        self.res2 = nn.Sequential(
            ConvBlock(512, 512),
            ConvBlock(512, 512)
        )
        
        # Global average pool → 1x1
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Returns: Logits of shape [batch_size, 20]
        """
        x = self.stem(x)        # 224 → 56
        
        x = self.layer1(x)      # 56 → 28
        x = self.res1(x) + x    # residual
        
        x = self.layer2(x)      # 28 → 14
        
        x = self.layer3(x)      # 14 → 7
        x = self.res2(x) + x    # residual
        
        x = self.pool(x)        # 7 → 1
        logits = self.classifier(x)
        
        return logits

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
