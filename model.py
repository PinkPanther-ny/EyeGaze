import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from torchvision.models import ResNet18_Weights


# Define the GazeNet model
class GazeNet(nn.Module):
    def __init__(self):
        super(GazeNet, self).__init__()
        self.backbone = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()  # Remove the final classification layer
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.backbone(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    # Instantiate the model
    model = GazeNet()

    # Create a dummy input with shape (32, 3, 224, 224)
    dummy_input = torch.rand((32, 3, 224, 224))

    # Pass the dummy input through the model
    output = model(dummy_input)

    # Verify the output shape is (32, 2)
    assert output.shape == (32, 2)
    print("Output shape:", output.shape)  # Output shape: (32, 2)
