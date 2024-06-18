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


if __name__ == '__main__':
    model_path = 'saved_models_hist/122_new.pth'
    # Instantiate the model
    model = GazeNet().to('cuda')
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(model)
    batch = 1
    dummy_input = torch.rand((batch, 3, 384, 384), dtype=torch.float32).to('cuda')

    # Export the model to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        "gazenet.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("Model has been converted to ONNX")
