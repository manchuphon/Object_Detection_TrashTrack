import numpy as np
import cv2
from ultralytics import YOLO

# กำหนดประเภทขยะและสีถังที่ต้องใช้
waste_categories = {
    "general": ("general - Put in the blue bin", (255, 0, 0)),  # น้ำเงิน
    "recyclable": ("recyclable - Put in the green bin", (0, 255, 0)),  # เขียว
    "hazardous": ("hazardous - Put in the red bin", (0, 0, 255)),  # แดง
    "organic": ("organic - Put in the yellow bin", (0, 255, 255)),  # เหลือง
    "no detection": ("Not detectable", (0, 0, 0))
}

# เปิดกล้อง
cap = cv2.VideoCapture(0)
model = YOLO('best.pt')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # ลดขนาดภาพลงเพื่อเพิ่มประสิทธิภาพ
    original_frame = frame.copy()  # เก็บภาพต้นฉบับ
    frame_resized = cv2.resize(frame, None, fx=0.5, fy=0.5)

    # เรียกใช้งานโมเดล YOLO
    results = model(frame_resized)  # เรียกโมเดล YOLO และรับผลลัพธ์การตรวจจับ

    # ตรวจสอบผลลัพธ์ที่ได้จาก YOLO
    for result in results:
        # สำหรับแต่ละผลลัพธ์ที่ได้จาก YOLO
        for box in result.boxes:
            # ดึงค่าพิกัดจากกรอบ
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # พิกัดกรอบ (xmin, ymin, xmax, ymax)
            cls_id = int(box.cls[0].item())  # ดึงค่า Class ID
            label = model.names[cls_id]  # แปลงเป็นชื่อวัตถุ

            # กำหนดประเภทขยะจาก label ที่ตรวจพบ
            if label in ["bottle", "light bulb"]:
                category = "recyclable"
            elif label in ["snacks", "ketchup"]:
                category = "general"
            elif label in ["dry battery"]:
                category = "hazardous"
            elif label in ["leaf"]:
                category = "organic"
            else:
                category = "no detection"

            msg, color = waste_categories[category]

            # ขยายขนาดกรอบให้ตรงกับขนาดภาพต้นฉบับ
            x1, y1, x2, y2 = int(x1 * 2), int(y1 * 2), int(x2 * 2), int(y2 * 2)

            # วาดกรอบรอบขยะบนภาพต้นฉบับ
            cv2.rectangle(original_frame, (x1, y1), (x2, y2), color, 2)

            # แสดงข้อความบนภาพ
            cv2.putText(original_frame, msg, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # แสดงภาพที่มีการตรวจจับ
    cv2.imshow('Waste Classification', original_frame)

    # ปิดโปรแกรมเมื่อกดปุ่ม 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# ปิดการเชื่อมต่อกับกล้องและปิดหน้าต่าง
cap.release()
cv2.destroyAllWindows()
