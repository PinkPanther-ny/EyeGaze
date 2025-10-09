import numpy as np
import onnxruntime as ort


class OnnxModel:
    def __init__(self, model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider']):
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def inference(self, input_data):
        input_data = input_data.numpy().astype(np.float32)
        prediction = self.session.run([self.output_name], {self.input_name: input_data})[0][0]
        return prediction
