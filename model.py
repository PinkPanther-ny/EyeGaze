import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from torchvision.models import ResNet50_Weights


# Define the GazeNet model
class GazeNet(nn.Module):
    def __init__(self):
        super(GazeNet, self).__init__()
        self.backbone = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()  # Remove the final classification layer

        self.fc0 = nn.Linear(2048, 512)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.backbone(x)
        x = F.relu(self.fc0(x))
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x


if __name__ == '__main__':
    # Instantiate the model
    model = GazeNet()

    dummy_input = torch.rand((32, 3, 480, 640))

    # Pass the dummy input through the model
    output = model(dummy_input)

    # Verify the output shape is (32, 2)
    assert output.shape == (32, 2)
    print("Output shape:", output.shape)  # Output shape: (32, 2)
