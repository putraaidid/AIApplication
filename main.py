import cv2
import os

# --- Load Resources ---
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
if os.path.exists('Trainer.yml'):
    recognizer.read('Trainer.yml')
else:
    print("Trainer.yml not found. Please run train_faces.py first.")
    exit()

# --- Load Names ---
name_map = {}
names_file = "names.txt"

if os.path.exists(names_file):
    with open(names_file, "r") as f:
        for line in f:
            # clean whitespace
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue

            parts = line.split(",")
            
            # We need at least 2 parts (ID and Name)
            if len(parts) >= 2:
                try:
                    # Try to convert the first part to a number
                    user_id = int(parts[0])
                    user_name = parts[1]
                    name_map[user_id] = user_name
                except ValueError:
                    # If it fails (like "conda activate..."), just skip this line
                    print(f"[WARNING] Skipping bad line in names.txt: {line}")
                    continue

print("Loaded users:", name_map)

while True:
    ret, frame = video.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    # --- UI Design (Background/Header) ---
    # Draw a top bar for "Attendance System" look
    cv2.rectangle(frame, (0, 0), (640, 80), (35, 35, 35), -1) # Dark Bar
    cv2.putText(frame, "ATTENDANCE SYSTEM", (180, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)

    for (x, y, w, h) in faces:
        # Predict the face
        serial, conf = recognizer.predict(gray[y:y+h, x:x+w])
        
        # If confidence is low (meaning high match accuracy in LBPH lower is better, usually < 100)
        if conf < 100:
            name_label = name_map.get(serial, "Unknown")
            accuracy_text = f"{round(100 - conf)}%"
            color = (0, 255, 0) # Green for match
        else:
            name_label = "Unknown"
            accuracy_text = f"  {round(100 - conf)}%"
            color = (0, 0, 255) # Red for unknown

        # --- Draw Fancy Box around Face ---
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Name Tag Background
        cv2.rectangle(frame, (x, y-40), (x+w, y), color, -1)
        cv2.putText(frame, name_label, (x+5, y-10), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255), 1)

    cv2.imshow("Attendance System", frame)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()