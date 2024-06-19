import torch.nn as nn
import torch.onnx
import torchvision.models
from torchvision.models import ViT_B_16_Weights


# Define the GazeNet model
class GazeNet(nn.Module):
    def __init__(self):
        super(GazeNet, self).__init__()
        self.backbone = torchvision.models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
        self.backbone.heads = nn.Identity()  # Remove the final classification layer

        self.head = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x
