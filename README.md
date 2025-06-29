# YOLO Object Detection Web Application

This is a Streamlit web application for real-time object detection using YOLO (You Only Look Once) model. The app can detect lipbalm and minifan objects and send email alerts when detections occur.

## Features

- 🔍 Real-time object detection using YOLO model
- 📹 Live camera feed with detection overlays
- 📧 Email alerts when objects are detected
- 📋 Real-time detection log
- 📊 Detection statistics
- ⚙️ Configurable confidence threshold
- 🎨 Modern web interface

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure you have the YOLO model files:**
   - Make sure the `models/` directory contains your trained YOLO model files (`last.pt` or `best.pt`)

## Usage

### Running the Streamlit Web App

1. **Start the Streamlit application:**
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Open your web browser:**
   - The app will automatically open in your default browser
   - Default URL: `http://localhost:8501`

### Using the Application

1. **Configuration (Sidebar):**
   - Click "🚀 Start Detection" to begin real-time detection
   - Adjust the confidence threshold slider as needed
   - Use "📧 Test Email" to verify email functionality

2. **Main Interface:**
   - **Left Panel:** Live camera feed with detection overlays
   - **Right Panel:** Detection log and statistics

3. **Email Alerts:**
   - The app automatically sends email alerts when objects are detected
   - Email settings are configured in the sidebar

## Configuration

### Email Settings

The email functionality uses Gmail SMTP. Update the following variables in `streamlit_app.py`:

```python
EMAIL_SENDER = "your-email@gmail.com"
EMAIL_RECEIVER = "recipient-email@gmail.com"
EMAIL_PASSWORD = "your-app-password"  # Gmail App Password
```

**Note:** For Gmail, you need to use an App Password, not your regular password. Enable 2-factor authentication and generate an App Password in your Google Account settings.

### Model Configuration

- **Model Path:** The app uses `models/last.pt` by default
- **Target Classes:** Currently configured for 'lipbalm' and 'minifan'
- **Confidence Threshold:** Adjustable via the web interface (0.1 - 1.0)

## File Structure

```
my_model/
├── streamlit_app.py      # Main Streamlit application
├── detection_utils.py    # YOLO detection utilities
├── email_utils.py        # Email functionality
├── requirements.txt      # Python dependencies
├── models/
│   ├── last.pt          # YOLO model file
│   └── best.pt          # Best model file
└── README.md            # This file
```

## Troubleshooting

### Camera Issues
- Ensure your webcam is connected and accessible
- Check if other applications are using the camera
- Try restarting the application

### Email Issues
- Verify your Gmail App Password is correct
- Ensure 2-factor authentication is enabled on your Gmail account
- Check your internet connection

### Model Loading Issues
- Ensure the model files exist in the `models/` directory
- Check that the model files are not corrupted
- Verify the model is compatible with the current Ultralytics version

## Development

### Adding New Object Classes

1. Update the `target_classes` set in `streamlit_app.py`:
   ```python
   target_classes = {'lipbalm', 'minifan', 'new_object'}
   ```

2. Retrain your YOLO model to include the new classes

### Customizing the Interface

The Streamlit app uses a responsive layout with:
- Sidebar for controls and settings
- Main area for video feed and logs
- Real-time updates using session state

## License

This project is for educational and personal use. Please ensure you comply with all applicable laws and regulations when using this application.

## Cloud Deployment
⚠️ **Note:** Camera access is limited in cloud environments. For full functionality, run locally.

## Local Setup
1. Clone and install dependencies
2. Add your YOLO model files to `models/` directory
3. Run: `streamlit run streamlit_app.py`

## Cloud Usage
1. Select "Image Upload" mode
2. Upload an image file (PNG, JPG, etc.)
3. Click "Detect Objects" to process
4. View results and detection boxes

## Local Usage
1. Select "Camera" mode for real-time detection  
2. Or use "Image Upload" mode for specific images


# YOLO Object Detection Web Application

This is a Streamlit web application for real-time object detection using YOLO (You Only Look Once) model. The app can detect lipbalm and minifan objects and send email alerts when detections occur.

## Features

- 🔍 Real-time object detection using YOLO model
- 📹 Live camera feed with detection overlays
- 📧 Email alerts when objects are detected
- 📋 Real-time detection log
- 📊 Detection statistics
- ⚙️ Configurable confidence threshold
- 🎨 Modern web interface

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure you have the YOLO model files:**
   - Make sure the `models/` directory contains your trained YOLO model files (`last.pt` or `best.pt`)

## Usage

### Running the Streamlit Web App

1. **Start the Streamlit application:**
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Open your web browser:**
   - The app will automatically open in your default browser
   - Default URL: `http://localhost:8501`

### Using the Application

1. **Configuration (Sidebar):**
   - Click "🚀 Start Detection" to begin real-time detection
   - Adjust the confidence threshold slider as needed
   - Use "📧 Test Email" to verify email functionality

2. **Main Interface:**
   - **Left Panel:** Live camera feed with detection overlays
   - **Right Panel:** Detection log and statistics

3. **Email Alerts:**
   - The app automatically sends email alerts when objects are detected
   - Email settings are configured in the sidebar

## Configuration

### Email Settings

The email functionality uses Gmail SMTP. Update the following variables in `streamlit_app.py`:

```python
EMAIL_SENDER = "your-email@gmail.com"
EMAIL_RECEIVER = "recipient-email@gmail.com"
EMAIL_PASSWORD = "your-app-password"  # Gmail App Password
```

**Note:** For Gmail, you need to use an App Password, not your regular password. Enable 2-factor authentication and generate an App Password in your Google Account settings.

### Model Configuration

- **Model Path:** The app uses `models/last.pt` by default
- **Target Classes:** Currently configured for 'lipbalm' and 'minifan'
- **Confidence Threshold:** Adjustable via the web interface (0.1 - 1.0)

## File Structure

```
my_model/
├── streamlit_app.py      # Main Streamlit application
├── detection_utils.py    # YOLO detection utilities
├── email_utils.py        # Email functionality
├── requirements.txt      # Python dependencies
├── models/
│   ├── last.pt          # YOLO model file
│   └── best.pt          # Best model file
└── README.md            # This file
```

## Troubleshooting

### Camera Issues
- Ensure your webcam is connected and accessible
- Check if other applications are using the camera
- Try restarting the application

### Email Issues
- Verify your Gmail App Password is correct
- Ensure 2-factor authentication is enabled on your Gmail account
- Check your internet connection

### Model Loading Issues
- Ensure the model files exist in the `models/` directory
- Check that the model files are not corrupted
- Verify the model is compatible with the current Ultralytics version

## Development

### Adding New Object Classes

1. Update the `target_classes` set in `streamlit_app.py`:
   ```python
   target_classes = {'lipbalm', 'minifan', 'new_object'}
   ```

2. Retrain your YOLO model to include the new classes

### Customizing the Interface

The Streamlit app uses a responsive layout with:
- Sidebar for controls and settings
- Main area for video feed and logs
- Real-time updates using session state

## License

This project is for educational and personal use. Please ensure you comply with all applicable laws and regulations when using this application.

## Cloud Deployment
⚠️ **Note:** Camera access is limited in cloud environments. For full functionality, run locally.

## Local Setup
1. Clone and install dependencies
2. Add your YOLO model files to `models/` directory
3. Run: `streamlit run streamlit_app.py`

## Cloud Usage
1. Select "Image Upload" mode
2. Upload an image file (PNG, JPG, etc.)
3. Click "Detect Objects" to process
4. View results and detection boxes

## Local Usage
1. Select "Camera" mode for real-time detection
2. Or use "Image Upload" mode for specific images 