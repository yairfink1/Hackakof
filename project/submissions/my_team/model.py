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
    Lightweight ResNet-9 model architecture for local training from scratch.
    """
    def __init__(self, num_classes: int = 20):
        super().__init__()
        
        # Prep
        self.prep = ConvBlock(3, 64)
        
        # Layer 1
        self.layer1 = ConvBlock(64, 128, pool=True)
        self.res1 = nn.Sequential(
            ConvBlock(128, 128),
            ConvBlock(128, 128)
        )
        
        # Layer 2
        self.layer2 = ConvBlock(128, 256, pool=True)
        
        # Layer 3
        self.layer3 = ConvBlock(256, 512, pool=True)
        self.res2 = nn.Sequential(
            ConvBlock(512, 512),
            ConvBlock(512, 512)
        )
        
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
        x = self.prep(x)
        
        x = self.layer1(x)
        x = self.res1(x) + x
        
        x = self.layer2(x)
        
        x = self.layer3(x)
        x = self.res2(x) + x
        
        x = self.pool(x)
        logits = self.classifier(x)
        
        return logits

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')