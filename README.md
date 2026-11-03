# Person Entry Recognition System

A modular, real-time Computer Vision and Deep Learning framework for person detection, tracking, line-crossing (entry/exit tripwire) detection, and face recognition.

---

## 📁 Project Architecture

```text
person_entry_recognition/
│
├── models/
│   ├── person_detection/
│   │   └── yolo11n.pt
│   │
│   ├── face_detection/
│   │   └── face_detector_weights.pth
│   │
│   └── face_recognition/
│       └── best_model.pth
│
├── config/
│   ├── config.yaml
│   └── camera_config.yaml
│
├── data/
│   ├── registered_faces/
│   ├── face_embeddings/
│   └── database/
│
├── captures/
│   ├── person_crops/
│   ├── face_crops/
│   └── aligned_faces/
│
├── src/
│   │
│   ├── camera/
│   │   ├── camera_stream.py
│   │   └── video_reader.py
│   │
│   ├── detection/
│   │   └── person_detector.py
│   │
│   ├── tracking/
│   │   └── person_tracker.py
│   │
│   ├── line_crossing/
│   │   ├── line_drawer.py
│   │   └── crossing_detector.py
│   │
│   ├── face/
│   │   ├── face_detector.py
│   │   ├── face_alignment.py
│   │   └── face_quality.py
│   │   └── face_capture.py
│   │
│   ├── recognition/
│   │   ├── embedding_extractor.py
│   │   ├── face_matcher.py
│   │   └── model.py
│   │
│   ├── database/
│   │   ├── db_connection.py
│   │   ├── embedding_store.py
│   │   └── entry_logs.py
│   │
│   ├── utils/
│   │   ├── image_utils.py
│   │   └── logger.py
│   │
│   └── pipeline.py
│
├── logs/
│
├── tests/
│   ├── test_detection.py
│   ├── test_tracking.py
│   ├── test_line_crossing.py
│   └── test_recognition.py
│
├── main.py
├── requirements.txt
├── .env
└── README.md
```

---

## 🚀 Quick Start Guide

### 1. Environment Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

### 2. Configuration

Edit `config/config.yaml` to adjust:
- Model paths & detection thresholds
- Tripwire entry line coordinates (`line_start`, `line_end`)
- Direction (`top_to_bottom`, `bottom_to_top`, `left_to_right`, `right_to_left`)
- Face quality thresholds

Edit `config/camera_config.yaml` to specify webcam device index or RTSP camera URL.

---

## 🧑‍💼 Registering Faces

Register a known individual to the gallery database:

```bash
python main.py --register "John Doe" --image "data/registered_faces/john_doe.jpg"
```

---

## 📹 Running the Pipeline

### From Default Webcam (0):
```bash
python main.py --source 0
```

### From Recorded Video File:
```bash
python main.py --source "path/to/cctv_footage.mp4"
```

### Headless Server Mode (No GUI display):
```bash
python main.py --source 0 --no-show
```

---

## 🧪 Running Unit Tests

Run test suites using pytest:

```bash
pytest tests/
```

---

## 📊 Database Logs

Entry and exit events are automatically recorded in `data/database/entry_system.db` with:
- `track_id`: Unique tracking ID
- `person_name`: Identified person or 'Unknown'
- `confidence`: Cosine similarity score
- `direction`: Direction of crossing (ENTRY / EXIT)
- `snapshot_path`: File path to cropped snapshot
- `timestamp`: UTC event timestamp























