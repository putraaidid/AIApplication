import cv2
import os

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Ensure the base directory exists
base_dir = "dataset_faces"
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

# --- User Input ---
user_id = input("Enter your ID (Number only, e.g. 1): ")
name = input("Enter your Name: ")

# Create a specific folder for this user: "dataset_faces/User_1_Putra"
user_folder_name = f"User_{user_id}_{name}"
user_path = os.path.join(base_dir, user_folder_name)

if not os.path.exists(user_path):
    os.makedirs(user_path)
    print(f"[INFO] Created new folder: {user_path}")

# Update names.txt (Appends new user if not already there)
# We read it first to check if ID exists to avoid duplicates
existing_ids = []
if os.path.exists("names.txt"):
    with open("names.txt", "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 1:
                existing_ids.append(parts[0])

if user_id not in existing_ids:
    with open("names.txt", "a") as f:
        f.write(f"{user_id},{name}\n")

count = 0
print("Please look at the camera...")

while True:
    ret, frame = video.read()
    if not ret: break
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        count += 1
        
        # Save image INSIDE the user's specific folder
        file_name_path = os.path.join(user_path, f"User.{user_id}.{count}.jpg")
        cv2.imwrite(file_name_path, gray[y:y+h, x:x+w])
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
        
    cv2.imshow("Register New Face", frame)
    
    k = cv2.waitKey(1)
    if count >= 100:
        break
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
print("Data Collection Complete!")