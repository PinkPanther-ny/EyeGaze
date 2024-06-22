import os
additional_paths = [r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib', r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin', r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\libnvvp']
filtered_paths = [p for p in os.environ.get('PATH').split(os.pathsep) if 'NVIDIA GPU Computing Toolkit' not in p]
filtered_paths.extend(additional_paths)
os.environ['PATH'] = os.pathsep.join(filtered_paths)

import argparse
import torch
import pycuda.autoinit
import tensorrt as trt
from vit import GazeNet
from config import TARGET_SIZE

def export_to_onnx(model, dummy_input, onnx_file_path):
    torch.onnx.export(
        model,
        dummy_input,
        onnx_file_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Model has been converted to ONNX at {onnx_file_path}")

def export_to_jit(model, torchscript_file_path):
    # Export the model to TorchScript
    scripted_model = torch.jit.script(model)
    torch.jit.save(scripted_model, torchscript_file_path)
    print(f"Model has been converted to TorchScript at {torchscript_file_path}")

def build_trt_engine(onnx_file_path, engine_file_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(flags=1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 31)  # 2GB
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.STRICT_TYPES)

        # Parse model file
        with open(onnx_file_path, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                raise RuntimeError("Failed to parse ONNX file.")

        # Create optimization profile
        profile = builder.create_optimization_profile()
        profile.set_shape('input', (1, 3, TARGET_SIZE, TARGET_SIZE), (1, 3, TARGET_SIZE, TARGET_SIZE), (1, 3, TARGET_SIZE, TARGET_SIZE))
        config.add_optimization_profile(profile)

        # Build the engine
        engine = builder.build_serialized_network(network, config)
        if engine is None:
            raise RuntimeError("Failed to build the engine")

        # Serialize the engine and save it to file
        with open(engine_file_path, 'wb') as f:
            f.write(engine)
        print(f"Model has been converted to TensorRT engine at {engine_file_path}")

def convert_model(model_path, choice):
    base_name = os.path.splitext(model_path)[0]
    onnx_file_path = f'{base_name}.onnx'
    trt_file_path = f'{base_name}.trt'
    torchscript_file_path = f'{base_name}.pt'

    # Instantiate the model
    model = GazeNet().to('cuda')
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Create a dummy input for model tracing
    dummy_input = torch.rand((1, 3, TARGET_SIZE, TARGET_SIZE), dtype=torch.float32).to('cuda')

    if choice == 1:
        export_to_onnx(model, dummy_input, onnx_file_path)
    elif choice == 2:
        export_to_onnx(model, dummy_input, onnx_file_path)
        build_trt_engine(onnx_file_path, trt_file_path)
    elif choice == 3:
        export_to_jit(model, torchscript_file_path)
    elif choice == 4:
        export_to_onnx(model, dummy_input, onnx_file_path)
        build_trt_engine(onnx_file_path, trt_file_path)
        export_to_jit(model, torchscript_file_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert model to different formats")
    parser.add_argument('-p', '--model_path', type=str, help='Path to the model file')
    parser.add_argument('-m', '--mode', type=int, choices=[1, 2, 3, 4], default=4, help='Conversion option: 1 - ONNX, 2 - ONNX and TensorRT, 3 - TorchScript, 4 - ONNX, TensorRT, and TorchScript')
    args = parser.parse_args()

    convert_model(args.model_path, args.mode)
