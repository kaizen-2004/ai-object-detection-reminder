import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import winsound
from ultralytics import YOLO
from tkinter.scrolledtext import ScrolledText
from datetime import datetime  # Add this at the top once


# === Load YOLOv8 Model ===
MODEL_PATH = r"C:\Users\Kaizen\Downloads\ai-object-detection\my_model\last.pt"
model = YOLO(MODEL_PATH)
labels = model.names
confidence_threshold = 0.5
TARGET_CLASSES = {'lipbalm', 'minifan'}

class YOLOApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Lipbalm & Minifan Detector")
        self.running = False
        self.previous_status = ""

        self.root.geometry("900x700")
        self.root.configure(bg="#2c3e50")

        # === Webcam Setup ===
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640)
        self.cap.set(4, 480)

        # === Top Frame (Video + Status) ===
        self.top_frame = tk.Frame(self.root, bg="#2c3e50")
        self.top_frame.pack(fill=tk.BOTH, expand=True)

        # === Bottom Frame (Buttons) ===
        self.bottom_frame = tk.Frame(self.root, bg="#34495e")
        self.bottom_frame.pack(fill=tk.X, pady=10)

        # === Layout Config for top_frame ===
        self.top_frame.columnconfigure(0, weight=3)  # Video
        self.top_frame.columnconfigure(1, weight=2)  # Logs
        self.top_frame.rowconfigure(0, weight=1)
        self.top_frame.rowconfigure(1, weight=0)

        # === Video Display ===
        self.video_label = tk.Label(self.top_frame, bg="#2c3e50")
        self.video_label.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

        # === Logs Panel ===
        self.log_box = ScrolledText(self.top_frame, width=40, height=20, font=("Courier", 10), bg="#1e272e", fg="white")
        self.log_box.grid(row=0, column=1, padx=10, pady=20, sticky="nsew")

        # === Status Label ===
        self.status_label = tk.Label(
            self.top_frame,
            text="Click 'Start Detection' to begin.",
            font=("Helvetica", 12, "bold"),
            anchor="center",
            justify="center",
            bg="#2c3e50", fg="yellow",
            wraplength=800
)
        self.status_label.grid(row=1, column=0, columnspan=2, pady=10)

        # === Bottom Frame (holds button container) ===
        self.bottom_frame = tk.Frame(self.root, bg="#34495e")
        self.bottom_frame.pack(fill=tk.X, pady=10)

        # === Sub-frame for centering buttons ===
        self.button_container = tk.Frame(self.bottom_frame, bg="#34495e")
        self.button_container.pack(anchor="center")  # Centers within bottom_frame

        # === Buttons ===
        tk.Button(self.button_container, text="Start Detection", font=("Helvetica", 12, "bold"),
            command=self.start_detection, width=15,
            bg="#27ae60", fg="white", activebackground="#2ecc71").pack(side=tk.LEFT, padx=20)

        
        tk.Button(self.button_container, text="Stop Detection", font=("Helvetica", 12, "bold"),
            command=self.stop_detection, width=15,
            bg="#f39c12", fg="white", activebackground="#f1c40f").pack(side=tk.LEFT, padx=20)

        tk.Button(self.button_container, text="Quit", font=("Helvetica", 12, "bold"),
            command=self.quit_app, width=10,
            bg="#c0392b", fg="white", activebackground="#e74c3c").pack(side=tk.LEFT, padx=20)



        self.update_frame()


    def start_detection(self):
        self.running = True
        self.status_label.config(text="Detecting...")

    def stop_detection(self):
        self.running = False
        self.status_label.config(text="Detection stopped.")

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.status_label.config(text="Failed to grab frame.")
            return

        current_status = "Neither lipbalm nor minifan detected."
        detected_targets = set()

        if self.running:
            results = model(frame, verbose=False)
            detections = results[0].boxes

            for det in detections:
                conf = det.conf.item()
                if conf < confidence_threshold:
                    continue

                class_id = int(det.cls.item())
                label = labels[class_id].lower()

                if label not in TARGET_CLASSES:
                    continue

                detected_targets.add(label)

                xmin, ymin, xmax, ymax = det.xyxy.cpu().numpy().squeeze().astype(int)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (xmin, ymin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if 'lipbalm' in detected_targets and 'minifan' in detected_targets:
                current_status = "Both lipbalm and minifan detected."
            elif 'lipbalm' in detected_targets:
                current_status = "Lipbalm detected, minifan not detected."
            elif 'minifan' in detected_targets:
                current_status = "Minifan detected, lipbalm not detected."

            if current_status != self.previous_status:
                winsound.Beep(1000, 300)
                self.previous_status = current_status

            self.status_label.config(text=current_status)
            
        # Log detection status

        timestamp = datetime.now().strftime('%H:%M:%S')
        self.log_box.insert(tk.END, f"[{timestamp}] {current_status}\n")
        self.log_box.see(tk.END)  # Auto-scroll to bottom

        # Display video feed in GUI
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img_pil)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(10, self.update_frame)

    def quit_app(self):
        self.cap.release()
        self.root.destroy()

# === Run GUI ===
if __name__ == "__main__":
    root = tk.Tk()
    root.attributes("-fullscreen", True)  # Enable full screen on launch
    app = YOLOApp(root)
    root.mainloop()
