# Deep Road Analysis & Scene Perception System 🚗🛣️

An advanced Computer Vision and Traffic Scene Intelligence system powered by a custom **YOLO** object detection model (`best.pt`).

The system performs deep inference on road environments, detects 10 key classes of road participants, quantifies traffic density and congestion, evaluates ego-vehicle collision hazards, and renders high-aesthetic futuristic Heads-Up Displays (HUDs).

---

## 🎯 Model Capabilities

The custom model detects **10 road scene categories**:
- **Vehicles**: `car`, `bus`, `truck`
- **Two-Wheelers**: `bike`, `motor`
- **Vulnerable Road Users (VRUs)**: `person`, `rider`
- **Traffic Control & Infrastructure**: `traffic light`, `traffic sign`
- **Rail Transit**: `train`

---

## 🧠 Deep Scene Analytics

Beyond bounding boxes, the system computes:
1. **Traffic Density Index**: Free Flow, Moderate, Heavy, or Congested based on vehicle volume.
2. **Ego-Vehicle Hazard Corridor**: Identifies vehicles and pedestrians directly in the forward driving path.
3. **Collision Proximity Warnings**: Flags critical and caution proximity hazards in real-time.
4. **Road Safety Score (0-100)**: Quantitative safety metric for the current driving environment.
5. **HUD Telemetry Overlay**: Futuristic semi-transparent telemetry bar showing live KPIs.

---

## 🚀 Quick Start

### 1. One-Click Windows Launcher
Double-click `run.bat` or run:
```bat
run.bat
```

### 2. Run via Command Line (`main.py`)
Activate the virtual environment:
```powershell
.\.venv\Scripts\Activate.ps1
```

Analyze the included sample road image:
```bash
python main.py --source sample_road.jpg
```

Analyze with interactive image preview window:
```bash
python main.py --source sample_road.jpg --view
```

Analyze a video file:
```bash
python main.py --source dashcam_video.mp4 --view
```

Run live on your webcam:
```bash
python main.py --source 0 --view
```

Batch analyze an entire folder of images:
```bash
python main.py --source path/to/images/
```

### 3. Launch Interactive Web Dashboard
```bash
python main.py --web
```
or:
```bash
streamlit run app.py
```

---

## 📁 Output Structure

When running analysis, all outputs are automatically saved to `output/`:
- `output/<name>_analyzed.jpg` (or `.mp4` for video): High-resolution media with HUD and telemetry.
- `output/<name>_analytics.json`: Machine-readable structured JSON report containing:
  - Scene dimensions
  - Total counts and category distributions
  - Proximity alerts & hazard severity
  - Exact bounding box coordinates and confidence levels

---

## 🛠️ Project Structure

```
deep road analysis/
├── .venv/                      # Isolated Python virtual environment
├── best.pt                     # Custom YOLO detection model
├── sample_road.jpg             # High-resolution realistic test road image
├── road_analyzer.py            # Core analytics & HUD rendering engine
├── main.py                     # CLI & batch processing pipeline
├── app.py                      # Interactive Streamlit web app
├── requirements.txt            # Locked project dependencies
├── run.bat                     # Windows batch launcher
├── README.md                   # Documentation
└── output/                     # Auto-generated analytics and annotated media
```
