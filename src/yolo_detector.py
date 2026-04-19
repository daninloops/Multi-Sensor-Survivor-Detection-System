import cv2
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path="yolov8n.pt"):
        try:
            self.model = YOLO(model_path)
            print("YOLOv8 model loaded successfully.")
        except Exception as e:
            print(f"ERROR: Failed to load YOLOv8 model: {e}")
            raise

    def detect_objects(self, frame):
        """
        Detects persons in the frame.
        Returns a dictionary with detection status and confidence.
        Also draws bounding boxes on the frame.
        """
        results = self.model(frame, verbose=False)
        
        detected = False
        max_conf = 0.0
        
        # Iterate over detections
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Class 0 is usually 'person' in COCO dataset
                cls = int(box.cls[0])
                if cls == 0:
                    conf = float(box.conf[0])
                    if conf > 0.5:
                        detected = True
                        if conf > max_conf:
                            max_conf = conf
                        
                        # Draw bounding box
                        x1, y1, x2, y2 = box.xyxy[0]
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, f"Person {conf:.2f}", (int(x1), int(y1) - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return {
            "detected": detected,
            "confidence": max_conf
        }
