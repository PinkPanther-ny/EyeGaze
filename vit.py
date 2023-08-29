import torch.nn as nn
import torch.onnx
import torchvision.models
from torchvision.models import ViT_B_16_Weights

# Define the GazeNet model
class GazeNet(nn.Module):
    def __init__(self):
        super(GazeNet, self).__init__()
        self.backbone = torchvision.models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.backbone.heads = nn.Identity()  # Remove the final classification layer

        self.head = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


if __name__ == '__main__':
    # Instantiate the model
    model = GazeNet().to('cuda')
    model.eval()
    print(model)
    batch = 16
    dummy_input = torch.rand((batch, 3, 224, 224)).to('cuda')

    # Pass the dummy input through the model
    output = model(dummy_input)

    # Verify the output shape is (batch, 2)
    assert output.shape == (batch, 2)
    print("Output shape:", output.shape)  # Output shape: (batch, 2)
