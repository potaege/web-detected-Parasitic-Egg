from flask import Flask, request, jsonify, render_template, send_from_directory
from ultralytics import YOLO
import cv2
import numpy as np
import os

app = Flask(__name__)

# Load YOLO model
model = YOLO("best.pt")

# Create a directory to store detected images
UPLOAD_FOLDER = "static/uploads"
DETECTED_FOLDER = "static/detections"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DETECTED_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template("index.html")  # Load UI from HTML file

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded!"})

    file = request.files['image']
    filename = file.filename
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    detected_image_path = os.path.join(DETECTED_FOLDER, filename)

    # Save the uploaded image
    file.save(image_path)

    # Load image using OpenCV
    img = cv2.imread(image_path)

    # Run YOLO inference
    results = model(img, conf=0.25)

    # Draw bounding boxes on the image
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_name = model.names[int(box.cls)]
            confidence = float(box.conf)

            # Draw rectangle and label
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{class_name} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the detected image
    cv2.imwrite(detected_image_path, img)

    # Extract detected objects
    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class": model.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": box.xyxy.tolist()[0]  # Convert to list
            })

    # Determine response message
    if detections:
        detected_classes = {d["class"] for d in detections}  # Get unique detected class names
        message = f"Detected: {', '.join(detected_classes)}"
    else:
        message = "Detected and found nothing."

    return jsonify({"detections": detections, "message": message, "image_url": f"/detections/{filename}"})

@app.route('/detections/<filename>')
def get_detected_image(filename):
    return send_from_directory(DETECTED_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
