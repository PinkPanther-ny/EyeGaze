import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from torchvision.models import ResNet101_Weights


# Define the GazeNet model
class GazeNet(nn.Module):
    def __init__(self):
        super(GazeNet, self).__init__()
        self.backbone = torchvision.models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
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

    import torch.onnx

    # Export the model
    torch.onnx.export(model,  # model being run
                      dummy_input,  # model input (or a tuple for multiple inputs)
                      "gaze_net.onnx",  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'])  # the model's output names
