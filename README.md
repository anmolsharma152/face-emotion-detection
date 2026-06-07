# 😐 Face Emotion Detection

Real-time facial emotion detection — runs fully on CPU, no GPU required. This tool acts as a rapid prototyping script to detect faces and classify human emotions in real-time.

## Features Added
- **Multi-face support**: Detects and labels emotions for multiple people in the frame simultaneously.
- **FPS Counter**: Tracks and displays the real-time frame rate.
- **Multiple Input Sources**: Supports live webcams, pre-recorded video files, and static images.
- **Output Saving**: Save the processed and annotated video or image directly to disk.
- **Performance**: DeepFace analysis runs every N frames (configurable); bounding boxes and labels render every frame from cached results using distance heuristics — keeping CPU usage manageable.

## Demo

```
[video feed]
┌──────────────┐
│ FPS: 28      │
│              │
│  😊 happy    │  ← dominant emotion label
│  [        ]  │  ← green bounding box
│              │
└──────────────┘
```

## Getting Started

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)

### Run

```bash
git clone https://github.com/anmolsharma152/face_emotion_detection.git
cd face_emotion_detection

# Run with default webcam
uv run main.py
```

> First run downloads DeepFace model weights (~100MB). TensorFlow takes ~8 seconds to initialise — this is expected.

Press `q` to quit the video window.

### Command-Line Arguments

You can customize the execution with the following flags:

- `--source`: Camera index (e.g., `0`) or path to a video/image file. (Default: `0`)
- `--image`: Flag to indicate that the source is a static image file.
- `--skip-frames`: Run the heavy DeepFace analysis every N frames. (Default: `5`)
- `--output`: Path to save the annotated output video/image (e.g., `output.mp4` or `result.jpg`).

**Examples:**

```bash
# Process a video file and save the output
uv run main.py --source input_video.mp4 --output output_video.mp4

# Process a static image
uv run main.py --source my_photo.jpg --image --output annotated_photo.jpg

# Increase performance by running emotion detection less frequently
uv run main.py --skip-frames 10
```

### Notes

- **No GPU needed** — runs on CPU via TensorFlow.
- **Wayland warning** is harmless — OpenCV defaults to XWayland on GNOME/Wayland.
- **CUDA error** is harmless — if there is no GPU, TensorFlow falls back to CPU automatically.

## Potential & Future Use Cases

Currently, this project acts as a standalone command-line visualization tool. However, its core logic can be extracted and integrated into broader systems for real-world applications:

1. **Retail & Customer Analytics**: Integrate into store cameras or interactive kiosks to gauge customer sentiment and reactions to products or advertisements in real-time.
2. **User Experience & Gaming**: Use as an API layer in games or educational software to adapt the difficulty or narrative based on the user's frustration or engagement levels.
3. **Automotive Systems**: Build into driver monitoring systems to detect signs of anger, distress, or fatigue, triggering safety alerts.
4. **Healthcare & Therapy**: Assist in telehealth applications to log patient moods over time or provide tools for emotional regulation exercises.

### Integration Options
To move beyond a standalone script, the `FaceEmotionDetector` class could be adapted into:
- A **REST API (FastAPI / Flask)** that accepts image uploads and returns JSON emotion data.
- A **WebSockets server** streaming live detection telemetry to a React/Vue frontend dashboard.
- A **Dockerized Microservice** deployed in a cloud pipeline (e.g., AWS/GCP) to process large batches of video files asynchronously.

## Stack

| Component              | Library             |
| ---------------------- | ------------------- |
| Face detection         | OpenCV Haar Cascade |
| Emotion classification | DeepFace + tf-keras |
| Camera capture         | OpenCV VideoCapture |
| Package manager        | uv                  |
