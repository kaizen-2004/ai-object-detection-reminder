# Deployment Guide for Streamlit Cloud

## Quick Fix for Deployment Issues

If you're getting dependency installation errors, try these steps:

### 1. Use Minimal Requirements
The current `requirements.txt` has been simplified to work better with Streamlit Cloud.

### 2. Alternative: Use Minimal App
If the main app still fails, use `streamlit_app_minimal.py` instead:
```bash
streamlit run streamlit_app_minimal.py
```

### 3. Manual Deployment Steps

1. **Push to GitHub** with these files:
   - `streamlit_app.py` (or `streamlit_app_minimal.py`)
   - `requirements.txt`
   - `packages.txt`
   - `.streamlit/config.toml`
   - `detection_utils.py`
   - `email_utils.py`

2. **Deploy on Streamlit Cloud**:
   - Connect your GitHub repository
   - Set main file path to: `streamlit_app.py`
   - Deploy

### 4. Troubleshooting

#### If OpenCV fails:
- The app will run in demo mode
- Image upload will still work
- Detection will be simulated

#### If YOLO fails:
- The app will run in demo mode
- Upload your model files to `models/` directory
- Detection will be simulated until model is available

#### If Email fails:
- The app will run without email functionality
- All other features will work normally

### 5. Local Testing

Test locally first:
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### 6. Model Files

After deployment, upload your YOLO model files:
- Place `last.pt` or `best.pt` in the `models/` directory
- The app will automatically detect and use them

## File Structure for Deployment

```
my_model/
├── streamlit_app.py          # Main app
├── streamlit_app_minimal.py  # Fallback app
├── detection_utils.py        # Detection functions
├── email_utils.py           # Email functions
├── requirements.txt         # Python dependencies
├── packages.txt            # System dependencies
├── .streamlit/
│   └── config.toml         # Streamlit config
├── models/
│   └── .gitkeep           # Keep directory structure
└── README.md              # Documentation
```

## Common Issues

1. **OpenCV Import Error**: Use minimal app version
2. **PyTorch Installation**: CPU version is specified
3. **Model Loading**: Upload model files after deployment
4. **Email Issues**: Check Gmail app password settings

## Support

If deployment still fails:
1. Check Streamlit Cloud logs
2. Try the minimal app version
3. Test locally first
4. Ensure all files are in the repository 