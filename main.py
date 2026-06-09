import cv2
from deepface import DeepFace
import time
import argparse
import sys
import numpy as np
import threading
import queue
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        self.tracked_faces = {}  # dict mapping face_id (int) -> dict: {'bbox': (x,y,w,h), 'emotion': str, 'missing_frames': int}
        self.next_face_id = 0

        # Background Worker Queues
        self.task_queue = queue.Queue()
        self.results_queue = queue.Queue()
        self.analyzing_ids = set()

        # Start background inference worker
        self.worker_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self.worker_thread.start()

        # FPS tracking
        self.prev_frame_time = 0
        self.new_frame_time = 0

    def _inference_worker(self):
        """Worker running in background to perform DeepFace analysis without blocking the main loop."""
        while True:
            task = self.task_queue.get()
            if task is None:
                break
            face_id, face_roi = task
            try:
                result = DeepFace.analyze(
                    face_roi,
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend=self.detector_backend,
                    silent=True
                )
                emotion = result[0]['dominant_emotion']
                self.results_queue.put((face_id, emotion))
            except Exception as e:
                logging.error(f"Error analyzing face ID {face_id}: {e}")
                self.results_queue.put((face_id, None))
            finally:
                self.task_queue.task_done()

    def _match_faces(self, new_faces):
        """Match newly detected bounding boxes with existing tracked faces using centroid distance."""
        tracked_centroids = {}
        for face_id, tf in self.tracked_faces.items():
            x, y, w, h = tf['bbox']
            tracked_centroids[face_id] = (x + w//2, y + h//2)

        matched_face_ids = set()
        unmatched_new_faces = []

        for bbox in new_faces:
            x, y, w, h = bbox
            new_center_x, new_center_y = x + w//2, y + h//2

            best_id = None
            min_dist = float('inf')

            for face_id, (tx, ty) in tracked_centroids.items():
                if face_id in matched_face_ids:
                    continue
                dist = (new_center_x - tx)**2 + (new_center_y - ty)**2

                # 10000 square pixels threshold (100 pixels distance)
                if dist < min_dist and dist < 10000:
                    min_dist = dist
                    best_id = face_id

            if best_id is not None:
                matched_face_ids.add(best_id)
                self.tracked_faces[best_id]['bbox'] = bbox
                self.tracked_faces[best_id]['missing_frames'] = 0
            else:
                unmatched_new_faces.append(bbox)

        # Increment missing_frames for unmatched tracked faces
        for face_id in list(self.tracked_faces.keys()):
            if face_id not in matched_face_ids:
                self.tracked_faces[face_id]['missing_frames'] += 1
                if self.tracked_faces[face_id]['missing_frames'] > 10:
                    self.analyzing_ids.discard(face_id)
                    del self.tracked_faces[face_id]

        # Register new faces
        for bbox in unmatched_new_faces:
            face_id = self.next_face_id
            self.next_face_id += 1
            self.tracked_faces[face_id] = {
                'bbox': bbox,
                'emotion': 'Detecting...',
                'missing_frames': 0
            }

    def process_frame(self, frame):
        self.frame_count += 1
        display_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate FPS
        self.new_frame_time = time.time()
        fps = 1 / (self.new_frame_time - self.prev_frame_time) if self.prev_frame_time > 0 else 0
        self.prev_frame_time = self.new_frame_time

        # Update tracked faces with async inference results
        while not self.results_queue.empty():
            try:
                face_id, emotion = self.results_queue.get_nowait()
                if face_id in self.tracked_faces:
                    if emotion is not None:
                        self.tracked_faces[face_id]['emotion'] = emotion
                self.analyzing_ids.discard(face_id)
            except queue.Empty:
                break

        # Fast face detection every frame
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        # Match detections to update tracked faces
        self._match_faces(faces)

        # Queue DeepFace analysis asynchronously
        is_analysis_frame = (self.frame_count % self.skip_frames == 0)

        for face_id, tf in self.tracked_faces.items():
            if tf['missing_frames'] > 0:
                continue

            needs_analysis = is_analysis_frame or tf['emotion'] == 'Detecting...'

            if needs_analysis and face_id not in self.analyzing_ids:
                x, y, w, h = tf['bbox']
                pad_x = int(w * 0.1)
                pad_y = int(h * 0.1)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(frame.shape[1], x + w + pad_x)
                y2 = min(frame.shape[0], y + h + pad_y)

                face_roi = frame[y1:y2, x1:x2]

                if face_roi.size > 0:
                    self.analyzing_ids.add(face_id)
                    self.task_queue.put((face_id, face_roi.copy()))

        # Rendering
        for face_id, tf in self.tracked_faces.items():
            if tf['missing_frames'] > 0:
                continue

            x, y, w, h = tf['bbox']
            emotion = tf['emotion']

            # Draw rectangle
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Draw text label
            label = f"#{face_id}: {emotion}"
            cv2.putText(
                display_frame,
                label,
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
        # Call process_frame to queue the task
        detector.process_frame(frame)
        # Block until the background queue has processed the image task
        detector.task_queue.join()
        # Call process_frame again to update tracked_faces and render output
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
