import cv2
import numpy as np
from PIL import Image
import os

path = 'dataset_faces'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")

def getImagesAndLabels(path):
    faceSamples = []
    ids = []

    # os.walk allows us to look into all subdirectories
    for root, dirs, files in os.walk(path):
        for file in files:
            # We only want image files
            if file.endswith(".jpg"):
                imagePath = os.path.join(root, file)
                
                try:
                    # Convert to grayscale
                    PIL_img = Image.open(imagePath).convert('L')
                    img_numpy = np.array(PIL_img, 'uint8')

                    # Parse ID from filename (User.1.50.jpg)
                    # We split by '.' and get the second element
                    id = int(os.path.split(imagePath)[-1].split(".")[1])
                    
                    faces = detector.detectMultiScale(img_numpy)

                    for (x, y, w, h) in faces:
                        faceSamples.append(img_numpy[y:y+h, x:x+w])
                        ids.append(id)
                except Exception as e:
                    print(f"[WARNING] Skipping {imagePath}: {e}")

    return faceSamples, ids

print("\n [INFO] Training faces. This might take a moment...")
faces, ids = getImagesAndLabels(path)

if len(ids) > 0:
    recognizer.train(faces, np.array(ids))
    recognizer.write('Trainer.yml')
    print(f"\n [SUCCESS] {len(np.unique(ids))} distinct face IDs trained.")
else:
    print("\n [ERROR] No images found. Did you run add_faces.py?")