"""
Object Detection Module
Includes YOLOv8 + Door Detection
"""

from ultralytics import YOLO
import cv2
from core.config import MODEL_PATH, CONFIDENCE_THRESHOLD, CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT
from core.utils import draw_bounding_box, detect_door_shapes


class ObjectDetector:
    def __init__(self):
        print(f"Loading YOLOv8 from {MODEL_PATH}...")
        self.model = YOLO(MODEL_PATH)
        
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        
        if not self.cap.isOpened():
            raise Exception("Camera error")
        
        print("✅ Camera initialized")

    def detect(self, frame):
        """Basic YOLO detection"""
        results = self.model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        
        detections = []
        annotated_frame = frame.copy()
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                bbox = [int(x1), int(y1), int(x2), int(y2)]
                
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = self.model.names[cls]
                
                detections.append((label, bbox))
                annotated_frame = draw_bounding_box(annotated_frame, bbox, label, conf)
        
        return detections, annotated_frame
    
    def detect_with_doors(self, frame):
        """
        YOLO detection + Door detection
        THIS IS THE METHOD YOU SHOULD USE
        """
        # Get YOLO detections
        detections, annotated_frame = self.detect(frame)
        
        # Add door detections
        door_boxes = detect_door_shapes(frame)
        
        for bbox in door_boxes:
            detections.append(("door", bbox))
            
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(annotated_frame, "door", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return detections, annotated_frame

    def get_frame(self):
        """Get camera frame"""
        return self.cap.read()

    def release(self):
        """Release camera"""
        self.cap.release()
        cv2.destroyAllWindows()
        print("Camera released")
