import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Load lightweight pre-trained YOLOv8 Nano model
@st_cache_model = None  # Caching handled inside Streamlit app

def load_yolo_model():
    # Downloads yolo11n.pt / yolov8n.pt lightweight weights automatically
    model = YOLO("yolov8n.pt")
    return model

def detect_room_objects(image: Image.Image, confidence_threshold: float = 0.35):
    """
    Detects indoor objects/furniture in an image using YOLOv8.
    Returns:
        annotated_img (Image): Image with drawn 2D bounding boxes.
        detections (list): List of dicts containing class name, box, and confidence.
    """
    model = load_yolo_model()
    
    # Convert PIL Image to OpenCV BGR format
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Run YOLOv8 inference
    results = model(img_bgr, conf=confidence_threshold)[0]

    detections = []
    annotated_bgr = results.plot()  # Draws bounding boxes on image
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    # Extract detected box coordinates and label metadata
    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]

        detections.append({
            "label": label,
            "confidence": round(conf, 2),
            "bbox": [int(coord) for coord in xyxy]
        })

    return Image.fromarray(annotated_rgb), detections