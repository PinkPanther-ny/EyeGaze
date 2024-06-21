import threading
import time
import keyboard
import numpy as np
import pyautogui


class MouseControllerThread(threading.Thread):
    def __init__(self, buffer_size=50, duration=0.03):
        super().__init__()
        self.gaze_buffer = []
        self.buffer_size = buffer_size
        self.duration = duration
        self.lock = threading.Lock()
        self.running = True

    def run(self):
        while self.running:
            if keyboard.is_pressed('LEFT ALT'):
                with self.lock:
                    if len(self.gaze_buffer) > 0:
                        avg_gaze = np.mean(np.array(self.gaze_buffer), axis=0)
                        pyautogui.moveTo(avg_gaze[0], avg_gaze[1], duration=self.duration)
            time.sleep(0.001) # Avoid thread race condition

    def update_gaze(self, gaze_x, gaze_y):
        self.gaze_buffer.append((gaze_x, gaze_y))
        if len(self.gaze_buffer) > self.buffer_size:
            self.gaze_buffer.pop(0)

    def stop(self):
        self.running = False
