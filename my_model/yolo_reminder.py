import os
import sys
import argparse
import glob
import time
import cv2
import numpy as np
import winsound  # For sound alerts on Windows
from ultralytics import YOLO

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='Path to YOLO model')
parser.add_argument('--source', required=True, help='Path to image/video/folder or USB index (e.g. "usb0")')
parser.add_argument('--thresh', default=0.5, help='Confidence threshold')
parser.add_argument('--resolution', default=None, help='Resolution WxH')
parser.add_argument('--record', action='store_true', help='Record output video')
args = parser.parse_args()

model_path = args.model
img_source = args.source
min_thresh = float(args.thresh)
user_res = args.resolution
record = args.record

if not os.path.exists(model_path):
    print('ERROR: Model path not found.')
    sys.exit(0)

model = YOLO(model_path, task='detect')
labels = model.names

img_ext = ['.jpg','.jpeg','.png','.bmp']
vid_ext = ['.avi','.mp4','.mkv','.mov']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext.lower() in img_ext:
        source_type = 'image'
    elif ext.lower() in vid_ext:
        source_type = 'video'
    else:
        print(f'Unsupported file type: {ext}')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
else:
    print('Invalid source. Use image, folder, video file, or usbN (e.g., usb0).')
    sys.exit(0)

resize = False
if user_res:
    resize = True
    resW, resH = map(int, user_res.split('x'))

if record:
    if source_type not in ['video', 'usb']:
        print('Recording only works with video or webcam.')
        sys.exit(0)
    if not user_res:
        print('Specify --resolution to record.')
        sys.exit(0)
    recorder = cv2.VideoWriter('demo1.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (resW, resH))

if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = [f for f in glob.glob(img_source + '/*') if os.path.splitext(f)[1].lower() in img_ext]
elif source_type in ['video', 'usb']:
    cap = cv2.VideoCapture(img_source if source_type == 'video' else usb_idx)
    
    # Attempt to increase FPS and set optimal resolution
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if user_res:
        cap.set(3, resW)
        cap.set(4, resH)

bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133),
               (88,159,106), (96,202,231), (159,124,168), (169,162,241),
               (98,118,150), (172,176,184)]

img_count = 0
frame_rate_buffer = []
fps_avg_len = 200
avg_frame_rate = 0

# To track change in detection state
previous_msg = ""

while True:
    t_start = time.perf_counter()

    if source_type in ['image', 'folder']:
        if img_count >= len(imgs_list):
            print('All images processed.')
            break
        frame = cv2.imread(imgs_list[img_count])
        img_count += 1
    else:
        ret, frame = cap.read()
        if not ret or frame is None:
            print('Stream ended or camera disconnected.')
            break

    if resize:
        frame = cv2.resize(frame, (resW, resH))

    results = model(frame, verbose=False)
    detections = results[0].boxes
    detected_classes = set()

    for i in range(len(detections)):
        xyxy_tensor = detections[i].xyxy.cpu().numpy().squeeze()
        xmin, ymin, xmax, ymax = xyxy_tensor.astype(int)
        classidx = int(detections[i].cls.item())
        classname = labels[classidx].lower()
        conf = detections[i].conf.item()

        if conf > min_thresh:
            detected_classes.add(classname)
            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            label = f'{classname}: {int(conf*100)}%'
            label_size, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, label_size[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin - label_size[1] - 10),
                          (xmin + label_size[0], label_ymin + baseLine - 10), color, cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    # Detection reminder message
    minifan_detected = 'minifan' in detected_classes
    lipbalm_detected = 'lipbalm' in detected_classes

    if minifan_detected and lipbalm_detected:
        current_msg = "Both minifan and lipbalm detected."
    elif minifan_detected:
        current_msg = "Minifan detected, lipbalm not detected."
    elif lipbalm_detected:
        current_msg = "Lipbalm detected, minifan not detected."
    else:
        current_msg = "Neither minifan nor lipbalm detected."

    # Show message on OpenCV window
    cv2.putText(frame, current_msg, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Play sound if status changed
    if current_msg != previous_msg:
        winsound.Beep(1000, 300)  # Frequency: 1000Hz, Duration: 300ms
        previous_msg = current_msg

    # FPS display
    cv2.putText(frame, f'FPS: {avg_frame_rate:.2f}', (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow('YOLO Detection', frame)
    if record:
        recorder.write(frame)

    key = cv2.waitKey(0 if source_type in ['image', 'folder'] else 5)
    if key == ord('q') or key == ord('Q'):
        break

    t_stop = time.perf_counter()
    frame_rate_calc = 1 / (t_stop - t_start)
    frame_rate_buffer.append(frame_rate_calc)
    if len(frame_rate_buffer) > fps_avg_len:
        frame_rate_buffer.pop(0)
    avg_frame_rate = np.mean(frame_rate_buffer)

# Cleanup
print(f'Average FPS: {avg_frame_rate:.2f}')
if source_type in ['video', 'usb']:
    cap.release()
if record:
    recorder.release()
cv2.destroyAllWindows()
