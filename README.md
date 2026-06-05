# 😐 Face Emotion Detection

Real-time facial emotion detection using a webcam — runs fully on CPU, no GPU required.

## How It Works

- **Face detection**: OpenCV Haar Cascade (`haarcascade_frontalface_default.xml`)
- **Emotion classification**: DeepFace with `opencv` backend — detects 7 emotions: `angry`, `disgust`, `fear`, `happy`, `sad`, `surprise`, `neutral`
- **Performance**: DeepFace analysis runs every 5 frames; bounding box and label render every frame from cached results — keeps CPU usage manageable

## Demo

```
[webcam feed]
┌──────────────┐
│              │  ← green bounding box
│  😊 happy    │  ← dominant emotion label
│              │
└──────────────┘
```

## Getting Started

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)
- Webcam

### Run

```bash
git clone https://github.com/anmolsharma152/face_emotion_detection.git
cd face_emotion_detection

uv run main.py
```

> First run downloads DeepFace model weights (~100MB). TensorFlow takes ~8 seconds to initialise — this is expected.

Press `q` to quit.

### Notes

- **No GPU needed** — runs on CPU via TensorFlow
- **Wayland warning** is harmless — OpenCV defaults to XWayland on GNOME/Wayland
- **CUDA error** is harmless — no GPU on this machine, TensorFlow falls back to CPU automatically

## Stack

| Component | Library |
|---|---|
| Face detection | OpenCV Haar Cascade |
| Emotion classification | DeepFace + tf-keras |
| Camera capture | OpenCV VideoCapture |
| Package manager | uv |

## Project Structure

```
face_emotion_detection/
├── main.py          # Single-file implementation
├── pyproject.toml   # Dependencies
└── uv.lock          # Locked dependency versions
```
