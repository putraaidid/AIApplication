import cv2
import matplotlib.pyplot as plt

def draw_boundary(img, classifier, scaleFactor=1.1, minNeighbors=5, color=(0, 255, 0), text="Face", padding=6, thickness=2):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
    coords = []
    for (x, y, w, h) in features:
        # draw an outer box with padding
        x1, y1 = max(0, x - padding), max(0, y - padding)
        x2, y2 = x + w + padding, y + h + padding
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # name the box inside the outer box
        label_pos = (x1 + 5, y1 + 20)  # slightly inside top-left
        cv2.putText(img, text, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

        coords = [x, y, w, h]
    return coords, img

def detect(img, faceCascade):
    color = {"red": (255, 0, 0), "blue": (0, 0, 255), "green": (0, 255, 0)}
    coords, img = draw_boundary(img, faceCascade, scaleFactor=1.1, minNeighbors=10, color=color['green'], text="Face")
    return img

if __name__ == "__main__":
    # ------- Load Cascade -------
    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # ------- Load Image -------
    imagePath = "imageFace.jpg"
    img = cv2.imread(imagePath)
    if img is None:
        print(f"Failed to read image at {imagePath}. Please check the path.")
    else:
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # ------- Detect Face in Image using classifier -------
        faces = face_classifier.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40)
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # ------- Show Image -------
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 5))
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.show()

    # ------- Webcam Detection -------
    video_capture = cv2.VideoCapture(0)

    def detect_bounding_box(frame):
        _, frame = draw_boundary(frame, face_classifier, scaleFactor=1.5, minNeighbors=5, color=(0,255,0), text="Face")
        return _

    if not video_capture.isOpened():
        print("Webcam not found or could not be opened.")
    else:
        try:
            while True:
                ret, frame = video_capture.read()
                if not ret:
                    break

                detect_bounding_box(frame)
                cv2.imshow("My Face Detection Project", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            video_capture.release()
            cv2.destroyAllWindows()
