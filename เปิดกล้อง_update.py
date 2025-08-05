import numpy as np
import cv2
from ultralytics import YOLO

#กำหนดประเภทขยะและสีถังที่ต้องใช้
waste_categories = {
    "general": ("general - Put in the blue bin", (255, 0, 0)),  # น้ำเงิน
    "recyclable": ("recyclable - Put in the green bin", (0, 255, 0)),  # เขียว
    "hazardous": ("hazardous - Put in the red bin", (0, 0, 255)),  # แดง
    "organic": ("organic - Put in the yellow bin", (0, 255, 255)),  # เหลือง
    "no detection": ("Not detectable", (0,0,0))
}

#เปิดกล้อง
cap = cv2.VideoCapture(0)
model = YOLO('best.pt')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, None, fx=1.5, fy=1.5)
    results = model.predict(frame, conf=0.6,iou=0.5)

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0].item())  # ดึงค่า Class ID
            label = model.names[cls_id]  # แปลงเป็นชื่อวัตถุ

            # กำหนดประเภทขยะจาก label ที่ตรวจพบ
            if label in ["bottle", "light bulb"]:
                category = "recyclable"
            elif label in ["snacks", "ketchup"]:
                category = "general"
            elif label in "dry battery":
                category = "hazardous"
            elif label in "leaf":
                category = "organic"
            else:
                category = "no detection"

            msg, color = waste_categories[category]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # วาดกรอบรอบขยะ
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # แสดงข้อความบนภาพ
            cv2.putText(frame, msg, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow('Waste Classification', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()