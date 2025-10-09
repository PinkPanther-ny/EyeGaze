import pathlib


def initialize_backend(model_name):
    """
    Initializes the inference backend based on the model file suffix.
    
    Args:
    - model_name (str): Path to the model file.
    """
    model_path = pathlib.Path(model_name)
    suffix = model_path.suffix.lower()

    if suffix == '.pt':
        from . import pth_infer
        return pth_infer.PthModel(model_name)
    elif suffix == '.onnx':
        from . import onnx_infer
        return onnx_infer.OnnxModel(model_name)
    elif suffix == '.trt':
        from . import trt_infer
        return trt_infer.TrtModel(model_name)
    else:
        raise ValueError(f"Unsupported model file format: {suffix}")
