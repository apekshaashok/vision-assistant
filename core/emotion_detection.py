"""
Emotion Detection Module for Vision Assistant
Uses DeepFace for facial emotion analysis (real-time, plug-and-play)
"""

import cv2
from deepface import DeepFace

class EmotionDetector:
    def __init__(self, camera_index=0):
        print("Initializing EmotionDetector...")
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise Exception("Camera error: Unable to access webcam.")
        print("✅ EmotionDetector ready")

    def get_frame(self):
        """Grab a frame from the camera."""
        ret, frame = self.cap.read()
        return ret, frame

    def detect_emotion(self, frame):
        """Run emotion detection on input frame."""
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            # Safe extraction for both list and dict results
            if isinstance(result, dict) and "dominant_emotion" in result:
                emotion = result["dominant_emotion"]
            elif isinstance(result, list) and len(result) > 0 and "dominant_emotion" in result[0]:
                emotion = result[0]["dominant_emotion"]
            else:
                emotion = "unknown"
        except Exception as e:
            print(f"⚠️ Emotion detection error: {e}")
            emotion = "unknown"
        return emotion

    def annotate_frame(self, frame, emotion):
        """Overlay detected emotion label on the frame."""
        cv2.putText(frame, f'Emotion: {emotion}', (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        return frame

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
        print("Camera released (EmotionDetector)")

# Usage demo
if __name__ == "__main__":
    detector = EmotionDetector()
    while True:
        ret, frame = detector.get_frame()
        if not ret:
            break
        emotion = detector.detect_emotion(frame)
        annotated = detector.annotate_frame(frame, emotion)
        cv2.imshow("Emotion Detection", annotated)
        print(f"Emotion: {emotion}")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    detector.release()
    print("✅ EmotionDetection demo complete!")
