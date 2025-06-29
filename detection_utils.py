import cv2
import os
import sys
from ultralytics import YOLO

def resource_path(relative_path):
    """Get absolute path to resource, works for PyInstaller and normal run."""
    try:
        base_path = getattr(sys, '_MEIPASS', os.path.abspath(os.path.dirname(__file__)))
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

MODEL_PATH = resource_path(r"models/last.pt")

# === Constants ===
CONFIDENCE_THRESHOLD = 0.5
TARGET_CLASSES = {'lipbalm', 'minifan'}

def load_model(model_path=None):
    if model_path is None:
        model_path = resource_path("models/last.pt")
    model = YOLO(model_path)
    labels = model.names
    return model, labels

def detect_objects(frame, model, labels, confidence_threshold, target_classes):
    results = model(frame, verbose=False)
    detections = results[0].boxes
    detected_targets = set()

    for det in detections:
        conf = det.conf.item()
        if conf < confidence_threshold:
            continue

        class_id = int(det.cls.item())
        label = labels[class_id].lower()

        if label not in target_classes:
            continue

        detected_targets.add(label)
        xmin, ymin, xmax, ymax = det.xyxy.cpu().numpy().squeeze().astype(int)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if 'lipbalm' in detected_targets and 'minifan' in detected_targets:
        status = "Both lipbalm and minifan detected."
    elif 'lipbalm' in detected_targets:
        status = "Lipbalm detected, minifan not detected."
    elif 'minifan' in detected_targets:
        status = "Minifan detected, lipbalm not detected."
    else:
        status = "Neither lipbalm nor minifan detected."

    return frame, list(detected_targets), status

