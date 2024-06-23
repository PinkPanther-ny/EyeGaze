import threading

import cv2


class CameraCaptureThread(threading.Thread):
    def __init__(self, camera_index=0):
        super().__init__()
        self.cap = cv2.VideoCapture(camera_index)
        self.frame = None
        self.running = True
        self.lock = threading.Lock()

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            return self.frame

    def stop(self):
        self.running = False
        self.cap.release()
