import cv2
from deepface import DeepFace
import time
import argparse
import sys
import numpy as np

class FaceEmotionDetector:
    def __init__(self, skip_frames=5, detector_backend='opencv'):
        self.skip_frames = skip_frames
        self.detector_backend = detector_backend

        # Load OpenCV Haar Cascade for fast bounding box detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # State
        self.frame_count = 0
        self.tracked_faces = [] # list of dicts: {'bbox': (x, y, w, h), 'emotion': 'Detecting...'}

        # FPS tracking
        self.prev_frame_time = 0
        self.new_frame_time = 0

    def _match_faces(self, new_faces):
        """Match newly detected bounding boxes with existing tracked faces to retain emotions."""
        matched_tracked_faces = []

        for (x, y, w, h) in new_faces:
            center_x, center_y = x + w//2, y + h//2
            best_emotion = "Detecting..."
            min_dist = float('inf')

            for tf in self.tracked_faces:
                tx, ty, tw, th = tf['bbox']
                t_center_x, t_center_y = tx + tw//2, ty + th//2
                dist = (center_x - t_center_x)**2 + (center_y - t_center_y)**2

                if dist < min_dist and dist < 10000: # Arbitrary threshold for distance
                    min_dist = dist
                    best_emotion = tf['emotion']

            matched_tracked_faces.append({'bbox': (x, y, w, h), 'emotion': best_emotion})

        self.tracked_faces = matched_tracked_faces

    def process_frame(self, frame):
        self.frame_count += 1
        display_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate FPS
        self.new_frame_time = time.time()
        fps = 1 / (self.new_frame_time - self.prev_frame_time) if self.prev_frame_time > 0 else 0
        self.prev_frame_time = self.new_frame_time

        # Fast face detection every frame
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        # Match with previous faces to keep emotions
        self._match_faces(faces)

        # DeepFace Emotion classification every N frames
        if self.frame_count % self.skip_frames == 0:
            for i, tf in enumerate(self.tracked_faces):
                x, y, w, h = tf['bbox']
                # Expand bounding box slightly for DeepFace
                pad_x = int(w * 0.1)
                pad_y = int(h * 0.1)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(frame.shape[1], x + w + pad_x)
                y2 = min(frame.shape[0], y + h + pad_y)

                face_roi = frame[y1:y2, x1:x2]

                if face_roi.size > 0:
                    try:
                        result = DeepFace.analyze(
                            face_roi,
                            actions=['emotion'],
                            enforce_detection=False,
                            detector_backend=self.detector_backend,
                            silent=True
                        )
                        self.tracked_faces[i]['emotion'] = result[0]['dominant_emotion']
                    except Exception as e:
                        pass

        # Rendering
        for tf in self.tracked_faces:
            x, y, w, h = tf['bbox']
            emotion = tf['emotion']
            
            # Draw rectangle
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Draw text
            cv2.putText(
                display_frame,
                emotion,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (36, 255, 12),
                2
            )
            
        # Draw FPS
        cv2.putText(
            display_frame, 
            f"FPS: {int(fps)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 
            1,
            (255, 0, 0),
            2
        )

        return display_frame

def main():
    parser = argparse.ArgumentParser(description="Face Emotion Detection")
    parser.add_argument("--source", type=str, default="0", help="Camera index (e.g., 0) or path to video/image file")
    parser.add_argument("--image", action="store_true", help="Set to true if source is an image file")
    parser.add_argument("--skip-frames", type=int, default=5, help="Run deepface every N frames")
    parser.add_argument("--output", type=str, default="", help="Path to save output video/image")
    args = parser.parse_args()

    detector = FaceEmotionDetector(skip_frames=args.skip_frames)

    if args.image:
        frame = cv2.imread(args.source)
        if frame is None:
            print(f"Error: Could not read image {args.source}")
            sys.exit(1)

        # Force emotion detection on the first frame
        detector.skip_frames = 1
        output_frame = detector.process_frame(frame)

        if args.output:
            cv2.imwrite(args.output, output_frame)
            print(f"Saved output to {args.output}")
        else:
            cv2.imshow("Emotion Detection", output_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        # Try parsing source as int (for webcam) or string (for video file)
        source = int(args.source) if args.source.isdigit() else args.source
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            print(f"Cannot open video source: {args.source}")
            sys.exit(1)

        out = None
        if args.output:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

        print("Starting video feed... Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            display_frame = detector.process_frame(frame)

            if out:
                out.write(display_frame)

            cv2.imshow("Emotion Detection", display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        if out:
            out.release()
            print(f"Saved output to {args.output}")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
