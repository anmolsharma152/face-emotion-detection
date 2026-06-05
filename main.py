import cv2
from deepface import DeepFace
import time

# Load face detection model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

frame_count = 0
# Store the last detected emotion to display during skipped frames
last_emotion = "Detecting..."
last_face_coords = None

print("Starting video feed... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    
    # Copy frame for display
    display_frame = frame.copy()
    
    # gray frame for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- HEAVY LIFTING (Run only every 5 frames) ---
    if frame_count % 5 == 0:
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        if len(faces) > 0:
            # Just take the largest face
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            x, y, w, h = faces[0]
            last_face_coords = (x, y, w, h)
            
            # Crop face for DeepFace
            face_roi = frame[y:y+h, x:x+w]
            
            try:
                # Use 'opencv' backend for speed instead of retinaface
                result = DeepFace.analyze(
                    face_roi, 
                    actions=['emotion'], 
                    enforce_detection=False, 
                    detector_backend='opencv' 
                )
                last_emotion = result[0]['dominant_emotion']
            except Exception as e:
                pass
        else:
            last_face_coords = None

    # --- RENDERING (Run every frame) ---
    if last_face_coords:
        x, y, w, h = last_face_coords
        # Draw rectangle
        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Draw text
        cv2.putText(
            display_frame, 
            last_emotion, 
            (x, y - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.9, 
            (36, 255, 12), 
            2
        )

    cv2.imshow("Emotion Detection", display_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
