from flask import Flask, request, jsonify, render_template, send_from_directory
from ultralytics import YOLO
import cv2
import os
import uuid

app = Flask(__name__)

# โหลด YOLO model
model = YOLO("best.pt")

# สร้างโฟลเดอร์สำหรับเก็บภาพที่ตรวจพบ
UPLOAD_FOLDER = "static/uploads"
DETECTED_FOLDER = "static/detections"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DETECTED_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template("index.html")  # โหลด UI จากไฟล์ HTML

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "ไม่มีการอัปโหลดภาพ!"})

    file = request.files['image']
    original_filename = file.filename

    # สร้างชื่อไฟล์ที่ไม่ซ้ำกันโดยการเพิ่ม UUID เข้ากับชื่อไฟล์เดิม
    unique_filename = f"{uuid.uuid4().hex}_{original_filename}"
    
    image_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    detected_image_path = os.path.join(DETECTED_FOLDER, unique_filename)

    # บันทึกไฟล์ภาพที่อัปโหลดด้วยชื่อไฟล์ที่ไม่ซ้ำกัน
    file.save(image_path)

    # โหลดภาพด้วย OpenCV
    img = cv2.imread(image_path)

    # รัน YOLO inference
    results = model(img, conf=0.342)

    # วาดกรอบสี่เหลี่ยมบนภาพ
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_name = model.names[int(box.cls)]
            confidence = float(box.conf)

            # วาดกรอบสี่เหลี่ยมและป้าย
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{class_name} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # บันทึกภาพที่ตรวจพบด้วยชื่อไฟล์ที่ไม่ซ้ำกัน
    cv2.imwrite(detected_image_path, img)

    # ดึงข้อมูลวัตถุที่ตรวจพบ
    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class": model.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": box.xyxy.tolist()[0]  # แปลงเป็นลิสต์
            })

    # กำหนดข้อความตอบกลับ
    if detections:
        detected_classes = {d["class"] for d in detections}  # ดึงชื่อคลาสที่ตรวจพบที่ไม่ซ้ำ
        message = f"Detected: {', '.join(detected_classes)}"
    else:
        message = "ไม่พบวัตถุที่ตรวจพบ"

    return jsonify({"detections": detections, "message": message, "image_url": f"/detections/{unique_filename}"})

@app.route('/detections/<filename>')
def get_detected_image(filename):
    return send_from_directory(DETECTED_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
