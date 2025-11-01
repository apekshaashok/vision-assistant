from ultralytics import YOLO
import cv2

# Load the action model (change filename if needed)
model = YOLO("models/action_yolov8.pt")

# Load your test image
img = cv2.imread("image.jpg")
if img is None:
    raise Exception("Could not read image.jpg")

# Detect actions
results = model(img)

# Draw results
for r in results:
    boxes = r.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        label = model.names[int(box.cls[0])]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        color = (0, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{label} ({conf:.2f})", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        print(f"Detected: {label} at {x1},{y1},{x2},{y2} (conf={conf:.2f})")

cv2.imshow("Action Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

