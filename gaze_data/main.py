import os
import sys
import time
import random
import tkinter as tk
from threading import Thread

import cv2
import keyboard
import mouse
import shortuuid
import win32con
import win32gui
from PIL import ImageTk, Image


screen_width = 1920
screen_height = 1080
take_photo_interval = 0.05

_last_take_time = 0
current_frame = None
current_mouse_loc = None


def get_random_id():
    return shortuuid.uuid().lower()[:6]


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


def set_click_through(hwnd):
    try:
        styles = win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT
        win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, styles)
        win32gui.SetLayeredWindowAttributes(hwnd, 0, 255, win32con.LWA_ALPHA)
    except Exception as e:
        print(e)


def get_mouse_loc():
    data = mouse.get_position()
    return f"{round(float(data[0] / screen_width), 3)}_{round(float(data[1] / screen_height), 3)}"


def convert_cv2_to_tkinter_image(cv2_img):
    cv2_img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cv2_img_rgb).convert("RGBA")
    
    # Create transparency
    alpha = 255  # transparency
    alpha_channel = Image.new('L', pil_img.size, alpha)
    pil_img.putalpha(alpha_channel)
    
    tk_img = ImageTk.PhotoImage(image=pil_img)
    return tk_img


def save_current_frame():
    global current_frame, current_mouse_loc
    if current_frame is not None and current_mouse_loc is not None:
        # Decide whether to save to train or val
        if random.random() < 0.9:
            subdir = 'train'
        else:
            subdir = 'val'

        filename = f"{output_dir}/{subdir}/{current_mouse_loc}_{get_random_id()}.jpg"
        cv2.imwrite(filename, current_frame)
        print(f"Saved: {filename}")


def capture():
    global _last_take_time, take_photo_interval
    if time.time() - _last_take_time > take_photo_interval:
        _last_take_time = time.time()
        save_current_frame()


def exit_app():
    os._exit(0)


def widget_follow_mouse(widget):
    while True:
        data = mouse.get_position()
        widget.place(
            relx=float(data[0] / screen_width),
            rely=float(data[1] / screen_height),
            anchor='center',
            width=22,
            height=22
        )
        time.sleep(1 / 120)


def center_crop_image(cv2_img):
    height, width, _ = cv2_img.shape
    min_dim = min(height, width)
    start_x = (width - min_dim) // 2
    start_y = (height - min_dim) // 2
    cropped_img = cv2_img[start_y:start_y + min_dim, start_x:start_x + min_dim]
    return cropped_img


def update_frame_and_mouse():
    global current_frame, current_mouse_loc, tk_img

    _, frame = camera.read()
    current_frame = frame
    current_mouse_loc = get_mouse_loc()
    
    # Center crop the image to a square
    cropped_frame = center_crop_image(frame)
    
    # Display the image on the canvas in the center of the grid
    tk_img = convert_cv2_to_tkinter_image(cropped_frame)
    c.itemconfig(image_item, image=tk_img)
    
    # Schedule the next update
    root.after(5, update_frame_and_mouse)


if __name__ == '__main__':
    camera = cv2.VideoCapture(0)
    output_dir = 'images'
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)

    def create_grid(event=None):
        spacing = 100
        w = c.winfo_width()  # Get current width of canvas
        h = c.winfo_height()  # Get current height of canvas
        c.delete('grid_line')  # Will only remove the grid_line

        # Creates all vertical lines at intervals of 50
        for i in range(0, w, spacing):
            c.create_line([(i, 0), (i, h)], tag='grid_line', fill='white')

        # Creates all horizontal lines at intervals of 50
        for i in range(0, h, spacing):
            c.create_line([(0, i), (w, i)], tag='grid_line', fill='white')


    root = tk.Tk()

    root.wm_attributes("-topmost", True)
    root.title("EyeGaze")
    root.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth(), root.winfo_screenheight()))
    root.config(bg='#000000')
    ico = ImageTk.PhotoImage(Image.open(resource_path('ico.ico')))
    root.iconphoto(False, ico)
    root.wm_attributes('-fullscreen', 'True')
    root.wm_attributes("-alpha", 0.85)

    c = tk.Canvas(root, bg='#333333')
    c.pack(fill=tk.BOTH, expand=True)
    c.bind('<Configure>', create_grid)

    # Set the photo image on the canvas
    tk_img = None
    image_item = c.create_image(screen_width//2, screen_height//2, anchor=tk.CENTER, image=tk_img)

    label = tk.Label(root, text=f'X', fg='blue', font=('helvetica', 16, 'bold'), justify=tk.CENTER)
    set_click_through(label.winfo_id())

    Thread(target=widget_follow_mouse, args=(label,)).start()

    # Start the update loop
    root.after(0, update_frame_and_mouse)

    keyboard.add_hotkey('SPACE', capture)
    keyboard.add_hotkey('ESC', exit_app)

    root.mainloop()
