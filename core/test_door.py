"""
Test door detection independently
"""

import cv2
import numpy as np
from core.utils import detect_door_shapes

# Open camera
cap = cv2.VideoCapture(0)

print("Testing door detection...")
print("Point camera at a door and press SPACE to test")
print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera error")
        break
    
    # Detect doors
    door_boxes = detect_door_shapes(frame)
    
    # Draw detected doors
    for i, bbox in enumerate(door_boxes):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Red box
        cv2.putText(frame, f"DOOR {i+1}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Show count
    cv2.putText(frame, f"Doors detected: {len(door_boxes)}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("Door Detection Test", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):  # Space bar
        print(f"\nDoors detected: {len(door_boxes)}")
        for i, bbox in enumerate(door_boxes):
            print(f"  Door {i+1}: {bbox}")

cap.release()
cv2.destroyAllWindows()
print("Test complete")
