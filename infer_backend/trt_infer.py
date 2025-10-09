import os

additional_paths = [r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib',
                    r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin',
                    r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\libnvvp']
filtered_paths = [p for p in os.environ.get('PATH').split(os.pathsep) if 'NVIDIA GPU Computing Toolkit' not in p]
filtered_paths.extend(additional_paths)
os.environ['PATH'] = os.pathsep.join(filtered_paths)

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit # noqa
import tensorrt as trt


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class TrtModel:
    def __init__(self, engine_path, max_batch_size=1, dtype=np.float32):
        self.engine_path = engine_path
        self.dtype = dtype
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(self.runtime, self.engine_path)
        self.max_batch_size = max_batch_size
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        self.context = self.engine.create_execution_context()

    @staticmethod
    def load_engine(trt_runtime, engine_path):
        trt.init_libnvinfer_plugins(None, "")
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return engine

    def allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.max_batch_size
            host_mem = cuda.pagelocked_empty(size, self.dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))

        return inputs, outputs, bindings, stream

    def inference(self, x: np.ndarray, batch_size=1):
        x = x.numpy().astype(self.dtype)
        np.copyto(self.inputs[0].host, x.ravel())

        for inp in self.inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)

        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream)

        self.stream.synchronize()
        out = [out.host.reshape(batch_size, -1) for out in self.outputs][0][0]
        return out


# Optional: Direct execution testing
if __name__ == "__main__":
    from augmentation import val_aug as transform
    import cv2
    import time

    batch_size = 1
    trt_engine_path = "gazenet.trt"  # Path to your TensorRT engine file
    model = TrtModel(trt_engine_path)

    cap = cv2.VideoCapture(0)
    pure_infer_time = 0
    for i in range(5000):
        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = transform(image=image)['image'].unsqueeze(0).numpy()

        t0_1 = time.time()
        result = model(input_tensor, batch_size)
        t = time.time() - t0_1
        pure_infer_time += t
        print(t)
