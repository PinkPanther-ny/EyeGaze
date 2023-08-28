import torch.nn as nn
import torch.onnx
import torchvision.models
from torchvision.models import ResNet101_Weights

def freeze_module(module, freeze=True):
    for param in module.parameters():
        param.requires_grad = not freeze

# Define the GazeNet model
class GazeNet(nn.Module):
    def __init__(self):
        super(GazeNet, self).__init__()
        self.backbone = torchvision.models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()  # Remove the final classification layer

        self.head = nn.Sequential(
            nn.Linear(2048, 512),
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

    def set_freeze(self, freeze=True):
        for i, child in enumerate(self.backbone.children(), 0):
            # Freeze conv5
            if i >= 7:
                freeze_module(child, freeze=freeze)

            # Freeze part of conv4
            if i == 6:
                for j, c in enumerate(child.children(), 0):
                    if j > 16:
                        freeze_module(c, freeze=freeze)



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

    # Export the model
    torch.onnx.export(model,  # model being run
                      dummy_input,  # model input (or a tuple for multiple inputs)
                      "gaze_net.onnx",  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'])  # the model's output names
