import imutils  #used for image processing like resizing,rotating,displaying...
import time
import cv2 #used for open cv
import csv
import os #used for directorys

cascade = 'haarcascade_frontalface_default.xml'     # use of a Haar Cascade Classifier for detecting faces in images or videos
detector = cv2.CascadeClassifier(cascade)

# Get the number of records to input
n = int(input("Enter the number of records you want to input: "))

dataset = 'dataset'

for _ in range(n):
    Name = str(input("Enter your Name: "))
    Roll_Number = int(input("Enter your Roll Number: "))

    sub_data = Name  #sub folder in dataset
    path = os.path.join(dataset, sub_data)

    if not os.path.isdir(path):
        os.mkdir(path)
        print(f"Directory created for {sub_data}")

    # Store the info in the CSV file
    info = [str(Name), str(Roll_Number)]
    with open('student.csv', 'a') as csvFile:
        write = csv.writer(csvFile)
        write.writerow(info)
    csvFile.close()

    
    print(f"Starting video stream for {Name}...")
    cam = cv2.VideoCapture(0)
    time.sleep(2.0) #with 2sec it capture 
    total = 0

    while total < 50:
        print(f"Capturing image {total + 1} for {Name}...")
        _, frame = cam.read()   #reading the face and storing in the frame
        img = imutils.resize(frame, width=400)  #resize
        
        rects = detector.detectMultiScale(
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), scaleFactor=1.1,
            minNeighbors=5, minSize=(30, 30)) #coordinates

        for (x, y, w, h) in rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            p = os.path.sep.join([path, "{}.png".format(
                str(total).zfill(5))])  #storing in png format
            cv2.imwrite(p, img)  #image save-imwrite
            total += 1

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):      #press q to quit
            break

    cam.release()

cv2.destroyAllWindows()
print("Video capture complete for all records.")
