from collections.abc import Iterable
import numpy as np
import imutils
import pickle
import time
import cv2
import csv
from datetime import datetime
import os  # For checking if the file already exists

# Function to flatten nested lists
def flatten(lis):
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:
            yield item

# File paths
embeddingFile = "output/embeddings.pickle"
embeddingModel = "openface_nn4.small2.v1.t7" #pytorch model
recognizerFile = "output/recognizer.pickle"
labelEncFile = "output/le.pickle"
conf = 0.75  # Set higher confidence threshold for more accuracy
attendance_file = 'attendance.csv'  # Path for attendance CSV

# Create the attendance file with headers if it doesn't exist
def create_attendance_file():
    if not os.path.exists(attendance_file):  # Check if the file already exists
        with open(attendance_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Name', 'Register Number', 'Date', 'Time'])  # Writing headers
            print(f"{attendance_file} created with headers.")

# Function to log attendance
def log_attendance(name, roll_number):
    with open(attendance_file, 'a', newline='') as f:
        writer = csv.writer(f)
        date_str = datetime.now().strftime("%Y-%m-%d")
        time_str = datetime.now().strftime("%H:%M:%S")
        writer.writerow([name, roll_number, date_str, time_str])
        print(f"Attendance logged for {name} (Roll: {roll_number})")

# Check if attendance for the person is already logged today
def already_logged(name, roll_number):
    date_str = datetime.now().strftime("%Y-%m-%d")
    with open(attendance_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == name and row[1] == roll_number and row[2] == date_str:
                return True
    return False

print("[INFO] loading face detector...")
prototxt = "model/deploy.prototxt"
model = "model/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt, model)

print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(embeddingModel)

recognizer = pickle.loads(open(recognizerFile, "rb").read())
le = pickle.loads(open(labelEncFile, "rb").read())

# Load the student CSV to verify against faces
studentData = []
with open('student.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        studentData.append(row)

# Create attendance file before starting the video stream
create_attendance_file()

print("[INFO] starting video stream...")
cam = cv2.VideoCapture(0)
time.sleep(1.0)
conf = 0.9  # Increased confidence threshold for detecting faces
min_proba = 0.90  # Minimum probability for recognizing face

# Continue from where the previous code was
while True:
    _, frame = cam.read()
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    # Create a blob for face detection
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    detector.setInput(imageBlob)
    detections = detector.forward()

    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > conf:  # Increased confidence threshold for face detection
            # Face detected with high confidence
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            if fW < 20 or fH < 20:
                continue

            # Generate face embedding
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # Predict the identity
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            # Set a threshold for prediction probability
            if proba >= min_proba:  # Only log if probability is higher than 90%
                # Look up the register number for the detected face
                Roll_Number = ""
                for row in studentData:
                    if name in row:
                        Roll_Number = row[1]  # Assume register number is in the second column

                text = "{} : {} : {:.2f}%".format(name, Roll_Number, proba * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

                # Log attendance if not already logged
                if Roll_Number and not already_logged(name, Roll_Number):
                    log_attendance(name, Roll_Number)

    # Display the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(10) & 0xFF

    if key == ord('q'):  # 'q' key to quit
        break

cam.release()
cv2.destroyAllWindows()
