import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import csv
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ---------------------------- Setup ----------------------------
cap = cv2.VideoCapture("../Videos/people.mp4")  # Change path if needed
model = YOLO("../Yolo-Weights/yolov8s.pt")       # Change path if needed
mask = cv2.imread("mask.png")                    # Change path if needed
imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)  # Optional

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
limitsUp = [103, 161, 296, 161]
limitsDown = [527, 489, 735, 489]
totalCountUp = []
totalCountDown = []

# Data Logging
log_data = []
net_people = 0
frame_rate = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(frame_rate * 2)  # Log every 5 minutes
frame_count = 0
start_time = datetime.now()

# ---------------------------- Main Loop ----------------------------
while True:
    success, img = cap.read()
    if not success:
        break

    imgRegion = cv2.bitwise_and(img, mask)
    img = cvzone.overlayPNG(img, imgGraphics, (730, 260))

    results = model(imgRegion, stream=True)
    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            if currentClass == "person" and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)

    # Draw lines
    cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 0, 255), 5)
    cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 0, 255), 5)

    for result in resultsTracker:
        x1, y1, x2, y2, id = map(int, result)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2

        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=10)
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if limitsUp[0] < cx < limitsUp[2] and limitsUp[1] - 15 < cy < limitsUp[1] + 15:
            if id not in totalCountUp:
                totalCountUp.append(id)
                net_people += 1
                cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255, 0), 5)

        if limitsDown[0] < cx < limitsDown[2] and limitsDown[1] - 15 < cy < limitsDown[1] + 15:
            if id not in totalCountDown:
                totalCountDown.append(id)
                net_people -= 1
                cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0), 5)

    # Display counts
    cv2.putText(img, str(len(totalCountUp)), (929, 345), cv2.FONT_HERSHEY_PLAIN, 5, (139, 195, 75), 7)
    cv2.putText(img, str(len(totalCountDown)), (1191, 345), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 230), 7)

    # Logging
    frame_count += 1
    if frame_count >= frame_interval:
        current_time = start_time + timedelta(seconds=cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
        log_data.append([current_time.strftime("%Y-%m-%d %H:%M:%S"), net_people])
        frame_count = 0

    # Show
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------------------- Save CSV & Chart ----------------------------
cap.release()
cv2.destroyAllWindows()

with open("people_log.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Time", "PeopleOn1stFloor"])
    writer.writerows(log_data)

# Plot chart
times = [row[0] for row in log_data]
counts = [row[1] for row in log_data]

plt.figure(figsize=(10, 5))
plt.plot(times, counts, marker='o', linestyle='-', color='blue')
plt.xticks(rotation=45)
plt.title("People on 1st Floor Over Time")
plt.xlabel("Time")
plt.ylabel("People Count")
plt.tight_layout()
plt.grid()
plt.savefig("people_chart.png")
plt.show()
