# YOLOv11 Image Detection (Single-Folder Flask App)

A minimal Flask web app that performs image detection using Ultralytics YOLOv11. Everything lives in one folder. The app installs Ultralytics directly from the GitHub repo you specified.

## Features

- Single-file app (`app.py`) with inline HTML (no templates/static folders)
- Uses `YOLO("yolo11n.pt")` and renders detections inline as base64 (no files saved)
- Installs `ultralytics` from GitHub: `https://github.com/ultralytics/ultralytics.git`

## Requirements

- Windows PowerShell
- Python 3.9–3.12
- Git installed (required for `pip` to install from GitHub URLs)

## Setup (PowerShell)

```powershell
# Navigate to the project folder
cd C:\Users\Asus\CascadeProjects\yolov11-image-detection

# Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies (Ultralytics from GitHub repo)
pip install -r requirements.txt

# If PyTorch install fails or is slow, try a CPU wheel:
# pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
```

## Run

```powershell
# Start the Flask app
python app.py
```

Then open http://127.0.0.1:5000 in your browser.

## Notes

- First run will download the `yolo11n.pt` weights automatically.
- You can switch to another model (e.g., `yolo11s.pt`) by editing `app.py`.
- For GPU acceleration, install the CUDA-compatible PyTorch per https://pytorch.org/get-started/locally/ before installing `ultralytics`.
