import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
from datetime import datetime, timedelta
import threading
from ultralytics import YOLO
import os
import sys
from detection_utils import detect_objects, load_model
from email_utils import send_email

# Page configuration
st.set_page_config(
    page_title="Group 6: Lipbalm and Minifan Detection with Reminder",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Email configuration
EMAIL_SENDER = "villasteve2004@gmail.com"
EMAIL_PASSWORD = "isdralnipvkyqaaw"  # App password

# Initialize session state
if 'detection_running' not in st.session_state:
    st.session_state.detection_running = False
if 'previous_status' not in st.session_state:
    st.session_state.previous_status = ""
if 'log_messages' not in st.session_state:
    st.session_state.log_messages = []
if 'model' not in st.session_state:
    st.session_state.model = None
if 'labels' not in st.session_state:
    st.session_state.labels = None
if 'email_enabled' not in st.session_state:
    st.session_state.email_enabled = True
if 'last_email_time' not in st.session_state:
    st.session_state.last_email_time = None
if 'email_interval_minutes' not in st.session_state:
    st.session_state.email_interval_minutes = 5
if 'email_receiver' not in st.session_state:
    st.session_state.email_receiver = "villasteve57@gmail.com"
if 'camera_available' not in st.session_state:
    st.session_state.camera_available = False
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'detection_mode' not in st.session_state:
    st.session_state.detection_mode = "upload"  # "upload" or "camera"

def load_yolo_model():
    """Load YOLO model"""
    if st.session_state.model is None:
        with st.spinner("Loading YOLO model..."):
            try:
                model, labels = load_model()
                st.session_state.model = model
                st.session_state.labels = labels
            except Exception as e:
                st.error(f"Failed to load model: {e}")
                return None, None
    return st.session_state.model, st.session_state.labels

def check_camera_availability():
    """Check if camera is available"""
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            return ret
        return False
    except Exception:
        return False

def log_status(status):
    """Add status message to log"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    log_entry = f"[{timestamp}] {status}"
    st.session_state.log_messages.append(log_entry)
    # Keep only last 50 messages
    if len(st.session_state.log_messages) > 50:
        st.session_state.log_messages.pop(0)

def can_send_email():
    """Check if enough time has passed since last email"""
    if not st.session_state.email_enabled:
        return False
    
    if st.session_state.last_email_time is None:
        return True
    
    time_since_last = datetime.now() - st.session_state.last_email_time
    interval = timedelta(minutes=st.session_state.email_interval_minutes)
    return time_since_last >= interval

def send_alert_email(message):
    """Send alert email with interval control"""
    if not can_send_email():
        log_status(f"Email skipped - too soon since last email (interval: {st.session_state.email_interval_minutes} min)")
        return
    
    success = send_email(
        subject="YOLO Detector Alert",
        body=f"{message} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        to_email=st.session_state.email_receiver,
        from_email=EMAIL_SENDER,
        app_password=EMAIL_PASSWORD
    )
    
    if success:
        st.session_state.last_email_time = datetime.now()
        log_status("Reminder sent successfully.")
    else:
        log_status("Reminder sending failed.")

def process_frame(frame, model, labels, confidence_threshold, target_classes):
    """Process a single frame for object detection"""
    frame, detected_targets, current_status = detect_objects(
        frame, model, labels, confidence_threshold, target_classes
    )
    return frame, detected_targets, current_status

def process_uploaded_image(image, model, labels, confidence_threshold, target_classes):
    """Process uploaded image for object detection"""
    # Convert PIL image to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Process the image
    processed_image, detected_targets, current_status = detect_objects(
        image_cv, model, labels, confidence_threshold, target_classes
    )
    
    # Convert back to RGB for display
    processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
    
    return processed_image_rgb, detected_targets, current_status

# Main app
def main():
    st.title("🔍 Group 6: Lipbalm and Minifan Detection with Reminder")
    st.markdown("---")
    
    # Check camera availability
    if not st.session_state.camera_available:
        st.session_state.camera_available = check_camera_availability()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Detection mode selection
        st.subheader("Detection Mode")
        detection_mode = st.radio(
            "Choose detection mode:",
            ["📷 Image Upload", "📹 Camera (if available)"],
            index=0 if st.session_state.detection_mode == "upload" else 1
        )
        
        if detection_mode == "📷 Image Upload":
            st.session_state.detection_mode = "upload"
        else:
            st.session_state.detection_mode = "camera"
        
        # Camera status
        if st.session_state.detection_mode == "camera":
            if st.session_state.camera_available:
                st.success("📹 Camera: Available")
            else:
                st.error("📹 Camera: Not Available")
                st.info("Camera access may be limited in cloud environments")
        
        # Detection controls
        st.subheader("Detection Controls")
        
        if st.session_state.detection_mode == "camera":
            start_button = st.button("🚀 Start Detection", type="primary", disabled=not st.session_state.camera_available)
            stop_button = st.button("⏹️ Stop Detection")
            
            if start_button:
                st.session_state.detection_running = True
                log_status("Detection started")
            
            if stop_button:
                st.session_state.detection_running = False
                log_status("Detection stopped")
        
        # Settings
        st.subheader("Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.5, 
            step=0.1
        )
        
        target_classes = {'lipbalm', 'minifan'}
        st.write("**Target Classes:**", ", ".join(target_classes))
        
        # Email settings
        st.subheader("📧 Email Settings")
        st.write(f"**Sender:** {EMAIL_SENDER}")
        
        # Customizable recipient email
        new_receiver = st.text_input(
            "Recipient Email Address",
            value=st.session_state.email_receiver,
            placeholder="Enter email address",
            help="Enter the email address where alerts should be sent"
        )
        
        # Update receiver if changed
        if new_receiver != st.session_state.email_receiver and new_receiver.strip():
            old_receiver = st.session_state.email_receiver
            st.session_state.email_receiver = new_receiver.strip()
            log_status(f"Email recipient changed from {old_receiver} to {new_receiver}")
        
        # Email control
        email_enabled = st.checkbox("Enable Email Alerts", value=st.session_state.email_enabled)
        if email_enabled != st.session_state.email_enabled:
            st.session_state.email_enabled = email_enabled
            if email_enabled:
                log_status("Email alerts enabled")
            else:
                log_status("Email alerts disabled")
        
        # Email interval setting
        if st.session_state.email_enabled:
            interval = st.slider(
                "Email Interval (minutes)", 
                min_value=1, 
                max_value=60, 
                value=st.session_state.email_interval_minutes, 
                step=1
            )
            if interval != st.session_state.email_interval_minutes:
                st.session_state.email_interval_minutes = interval
                log_status(f"Email interval changed to {interval} minutes")
            
            # Show time since last email
            if st.session_state.last_email_time:
                time_since = datetime.now() - st.session_state.last_email_time
                minutes_since = int(time_since.total_seconds() / 60)
                st.write(f"**Last email:** {minutes_since} minutes ago")
            else:
                st.write("**Last email:** Never sent")
        
        # Test email button
        if st.button("📧 Test Email"):
            if not st.session_state.email_receiver.strip():
                st.error("Please enter a valid email address")
            else:
                with st.spinner("Sending test email..."):
                    success = send_email(
                        subject="YOLO Detector Test",
                        body="This is a test email from the Group 6 - Detection and Reminder System.",
                        to_email=st.session_state.email_receiver,
                        from_email=EMAIL_SENDER,
                        app_password=EMAIL_PASSWORD
                    )
                    if success:
                        st.success("Test email sent successfully!")
                        st.session_state.last_email_time = datetime.now()
                    else:
                        st.error("Failed to send test email")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.session_state.detection_mode == "upload":
            st.subheader("📷 Image Upload Detection")
            
            # Image upload
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
                help="Upload an image to detect objects"
            )
            
            if uploaded_file is not None:
                # Display original image
                image = Image.open(uploaded_file)
                st.image(image, caption="Original Image", use_container_width=True)
                
                # Load model
                model, labels = load_yolo_model()
                
                if model is None:
                    st.error("❌ Failed to load YOLO model. Please check if model files are available.")
                else:
                    # Process button
                    if st.button("🔍 Detect Objects", type="primary"):
                        with st.spinner("Processing image..."):
                            # Process the uploaded image
                            processed_image, detected_targets, current_status = process_uploaded_image(
                                image, model, labels, confidence_threshold, target_classes
                            )
                            
                            # Display processed image
                            st.image(processed_image, caption="Processed Image with Detections", use_container_width=True)
                            
                            # Log the detection
                            log_status(current_status)
                            
                            # Send email if objects detected
                            if 'detected' in current_status.lower():
                                if st.session_state.email_enabled:
                                    threading.Thread(target=send_alert_email, args=(current_status,)).start()
                            
                            # Show detection results
                            if detected_targets:
                                st.success(f"✅ Detected: {', '.join(detected_targets)}")
                            else:
                                st.info("ℹ️ No target objects detected")
            
            else:
                st.info("📤 Please upload an image to start detection")
        
        else:  # Camera mode
            st.subheader("📹 Live Detection")
            
            # Video feed placeholder
            video_placeholder = st.empty()
            
            # Status display
            status_placeholder = st.empty()
            
            # Load model
            model, labels = load_yolo_model()
            
            if model is None:
                st.error("❌ Failed to load YOLO model. Please check if model files are available.")
                return
            
            # Camera capture
            if st.session_state.camera_available:
                cap = cv2.VideoCapture(0)
                cap.set(3, 640)
                cap.set(4, 480)
                
                if not cap.isOpened():
                    st.error("❌ Cannot open camera")
                    return
                
                # Detection loop
                while st.session_state.detection_running:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("❌ Failed to grab frame")
                        break
                    
                    # Process frame
                    frame, detected_targets, current_status = process_frame(
                        frame, model, labels, confidence_threshold, target_classes
                    )
                    
                    # Update status if changed
                    if current_status != st.session_state.previous_status:
                        log_status(current_status)
                        if 'detected' in current_status.lower():
                            # Send email in background thread with interval control
                            threading.Thread(target=send_alert_email, args=(current_status,)).start()
                        st.session_state.previous_status = current_status
                    
                    # Display frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(frame_rgb, channels="RGB")
                    
                    # Update status
                    if 'detected' in current_status.lower():
                        status_placeholder.error(f"🚨 {current_status}")
                    else:
                        status_placeholder.info(f"ℹ️ {current_status}")
                    
                    time.sleep(0.1)  # Small delay to prevent overwhelming the UI
                
                # Release camera when stopped
                if not st.session_state.detection_running:
                    cap.release()
                    video_placeholder.info("📹 Camera stopped. Click 'Start Detection' to begin.")
                    status_placeholder.info("⏸️ Detection paused")
            else:
                st.warning("⚠️ Camera not available in this environment. This app requires camera access for object detection.")
                st.info("💡 Try running this app locally for full camera functionality, or use Image Upload mode.")
    
    with col2:
        st.subheader("📋 Detection Log")
        
        # Log display
        log_container = st.container()
        with log_container:
            for message in reversed(st.session_state.log_messages[-20:]):  # Show last 20 messages
                st.text(message)
        
        # Clear log button
        if st.button("🗑️ Clear Log"):
            st.session_state.log_messages.clear()
            st.rerun()
        
        # Statistics
        st.subheader("📊 Statistics")
        if st.session_state.log_messages:
            detection_count = sum(1 for msg in st.session_state.log_messages if 'detected' in msg.lower())
            email_count = sum(1 for msg in st.session_state.log_messages if 'Reminder sent successfully' in msg)
            st.metric("Total Detections", detection_count)
            st.metric("Emails Sent", email_count)
            st.metric("Log Messages", len(st.session_state.log_messages))
        else:
            st.info("No detection data yet")

if __name__ == "__main__":
    main() 