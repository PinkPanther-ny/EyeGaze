import torch


class PthModel:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()  # Set the model to evaluation mode

    def inference(self, input_data):
        input_data = input_data.to(self.device)
        with torch.no_grad():
            prediction = self.model(input_data)
        return prediction.cpu().numpy()[0]  # Convert to numpy array for compatibility
