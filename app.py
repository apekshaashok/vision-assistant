import sys
import traceback
import math

# === WEB LOGGER ADDITION ===
class WebLogger:
    def __init__(self, log_file="web_output.log"):
        self.log_file = log_file
    def write(self, msg):
        with open(self.log_file, "a") as f:
            f.write(msg)
        sys.__stdout__.write(msg)
    def flush(self):
        pass
sys.stdout = WebLogger()
# === END WEB LOGGER ===

try:
    from core.detection import ObjectDetector
    from core.narration import Narrator
    from core.voice_control import VoiceController
    from core.ocr import TextReader
    from core.utils import (
        generate_spatial_description,
        get_position_info,
        check_command,
        generate_object_query_response,
    )
    from core.config import (
        DESCRIBE_COMMANDS, REPEAT_COMMANDS, EXIT_COMMANDS, WINDOW_NAME,
        OBJECT_QUERY_KEYWORDS, QUERYABLE_OBJECTS,
        CAMERA_INDEX, EMOTION_DETECTION_ENABLED
    )
    from core.emotion_detection import EmotionDetector
    import cv2
    import threading
except Exception as e:
    print(f"❌ Import error: {e}")
    traceback.print_exc()
    sys.exit(1)

def calc_center_angle(bbox, frame_width, hfov_deg=60):
    x1, y1, x2, y2 = bbox
    obj_center_x = (x1 + x2) / 2
    rel_center = (obj_center_x - frame_width / 2) / (frame_width / 2)
    angle = rel_center * (hfov_deg / 2)
    return angle

def estimate_distance_to_object(bbox, known_width=90, focal_length=580):
    x1, y1, x2, y2 = bbox
    width_pixels = abs(x2 - x1)
    if width_pixels <= 0:
        return None
    distance_cm = (known_width * focal_length) / width_pixels
    return distance_cm / 100  # meters

class VisionAssistantApp:
    def __init__(self):
        print("\n=== Vision Assistant: Starting ===\n")
        try:
            print("Initializing ObjectDetector...")
            self.detector = ObjectDetector()
            print("✅ ObjectDetector ready")
            print("Initializing Narrator...")
            self.narrator = Narrator()
            print("✅ Narrator ready")
            print("Initializing VoiceController...")
            self.voice_ctrl = VoiceController()
            print("✅ VoiceController ready")
            print("Initializing TextReader...")
            self.text_reader = TextReader()
            print("✅ TextReader ready")
            # EMOTION DETECTION
            if EMOTION_DETECTION_ENABLED:
                print("Initializing EmotionDetector...")
                self.emotion_detector = EmotionDetector(camera_index=CAMERA_INDEX)
                print("✅ EmotionDetector ready")
            else:
                self.emotion_detector = None
            self.running = True
            self.last_description = ""
            self.last_emotion = None
            print("✅ Initialization complete!\n")
        except Exception as e:
            print(f"❌ Init error: {e}")
            traceback.print_exc()
            sys.exit(1)
    def manual_controls(self):
        print("\n" + "="*60)
        print("VISION ASSISTANT v1.0")
        print("="*60)
        print("  [D] - Describe scene (includes doors!)")
        print("  [T] - Read text (OCR)")
        print("  [E] - Detect emotion (NEW!)")
        print("  [R] - Repeat last")
        print("  [Q] - Quit")
        print("\n  Voice Commands:")
        print("    'describe' - Full scan with doors")
        print("    'where is the door' - Find door")
        print("    'where is [object]' - Find object")
        print("    'read text' - OCR")
        print("    'repeat' - Repeat")
        print("    'emotion' - Detect emotion")
        print("    'stop' - Quit")
        print("="*60 + "\n")

    def voice_listener(self):
        while self.running:
            try:
                command = self.voice_ctrl.listen()
                if command is None:
                    continue
                print(f"🎤 '{command}'")
                if check_command(command, DESCRIBE_COMMANDS):
                    self.describe_scene()
                elif "read" in command or "text" in command:
                    self.read_text()
                elif check_command(command, REPEAT_COMMANDS):
                    self.repeat_description()
                elif check_command(command, EXIT_COMMANDS):
                    self.narrator.narrate("Goodbye!")
                    self.running = False
                    break
                elif self.is_object_query(command):
                    object_name = self.extract_object_name(command)
                    if object_name:
                        self.find_object(object_name)
                elif "emotion" in command and self.emotion_detector:
                    self.detect_emotion()
            except Exception as e:
                print(f"❌ Voice error: {e}")

    def is_object_query(self, command):
        return any(kw in command.lower() for kw in OBJECT_QUERY_KEYWORDS)

    def extract_object_name(self, command):
        cmd = command.lower()
        for obj in QUERYABLE_OBJECTS:
            if obj in cmd:
                return obj
        words = cmd.split()
        skip = ["is", "the", "a", "an", "where", "find", "locate", "there", "see", "can", "you", "do"]
        for word in words:
            if word not in skip and len(word) > 2:
                return word
        return None

    def find_object(self, object_name):
        try:
            print(f"\n🔍 Searching: {object_name}...")
            ret, frame = self.detector.get_frame()
            if not ret:
                self.narrator.narrate("Camera error.")
                return
            detections, annotated_frame = self.detector.detect_with_doors(frame)
            h, w = frame.shape[:2]
            found = False
            for (label, bbox) in detections:
                if label == "door" and object_name == "door":
                    angle = calc_center_angle(bbox, w)
                    distance_m = estimate_distance_to_object(bbox)
                    dist_text = f"{distance_m:.1f} meters" if distance_m else "unknown distance"
                    direction = "right" if angle > 0 else "left"
                    angle_deg = abs(int(round(angle)))
                    angle_phrase = "almost in front of you" if angle_deg < 7 else f"turn {angle_deg} degrees {direction}"
                    spoken = f"Door detected, {dist_text} ahead. To face the door, {angle_phrase}."
                    print(f"📢 {spoken}")
                    self.narrator.narrate(spoken)
                    found = True
                    break
            if not found:
                response = generate_object_query_response(object_name, detections, w, h)
                print(f"📢 {response}")
                self.narrator.narrate(response)
            cv2.imshow(WINDOW_NAME, annotated_frame)
            cv2.waitKey(700)
            print("✅ Done\n")
        except Exception as e:
            print(f"❌ Error: {e}")
            traceback.print_exc()

    def describe_scene(self):
        try:
            print("\n🔍 Scanning...")
            ret, frame = self.detector.get_frame()
            if not ret:
                self.narrator.narrate("Camera error.")
                return
            detections, annotated_frame = self.detector.detect_with_doors(frame)
            if not detections:
                desc = "I don't see any objects nearby."
            else:
                h, w = frame.shape[:2]
                spatial = []
                for label, bbox in detections:
                    direction, distance = get_position_info(bbox, w, h)
                    spatial.append((label, direction, distance))
                desc = generate_spatial_description(spatial)
            print(f"📢 {desc}")
            self.last_description = desc
            self.narrator.narrate(desc)
            cv2.imshow(WINDOW_NAME, annotated_frame)
            cv2.waitKey(300)
            print("✅ Done\n")
        except Exception as e:
            print(f"❌ Error: {e}")
            traceback.print_exc()

    def read_text(self):
        try:
            print("\n📖 Reading...")
            ret, frame = self.detector.get_frame()
            if not ret:
                self.narrator.narrate("Camera error.")
                return
            texts, annotated = self.text_reader.read_text(frame, preprocess=False)
            output = self.text_reader.format_text_output(texts)
            if texts:
                print(f"  • {texts}")
            print(f"📢 {output}")
            self.narrator.narrate(output)
            cv2.imshow(WINDOW_NAME, annotated)
            cv2.waitKey(500)
            print("✅ Done\n")
        except Exception as e:
            print(f"❌ Error: {e}")

    def detect_emotion(self):
        try:
            if not self.emotion_detector:
                print("⚠️ Emotion detection is not enabled.")
                return
            print("\n😊 Detecting emotion...")
            ret, frame = self.emotion_detector.get_frame()
            if not ret:
                self.narrator.narrate("Camera error.")
                return
            emotion = self.emotion_detector.detect_emotion(frame)
            annotated = self.emotion_detector.annotate_frame(frame, emotion)
            out_str = f"You look {emotion}."
            print(f"📢 {out_str}")
            if emotion != self.last_emotion:
                self.narrator.narrate(out_str)
                self.last_emotion = emotion
            cv2.imshow(WINDOW_NAME, annotated)
            cv2.waitKey(500)
            print("✅ Done\n")
        except Exception as e:
            print(f"❌ Error: {e}")

    def repeat_description(self):
        if self.last_description:
            print(f"\n🔁 {self.last_description}")
            self.narrator.narrate(self.last_description)
        else:
            msg = "No previous description."
            print(f"\n⚠️ {msg}")
            self.narrator.narrate(msg)

    def run(self):
        try:
            self.manual_controls()
            print("Starting voice...")
            voice_thread = threading.Thread(target=self.voice_listener, daemon=True)
            voice_thread.start()
            print("✅ Voice active\n")
            while self.running:
                ret, frame = self.detector.get_frame()
                if not ret:
                    break
                cv2.putText(frame, "D: Scan | T: Text | E: Emotion | R: Repeat | Q: Quit",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n👋 Quitting...")
                    self.narrator.narrate("Goodbye!")
                    self.running = False
                elif key == ord('d'):
                    self.describe_scene()
                elif key == ord('t'):
                    self.read_text()
                elif key == ord('r'):
                    self.repeat_description()
                elif key == ord('e') and self.emotion_detector:
                    self.detect_emotion()
            self.detector.release()
            if self.emotion_detector:
                self.emotion_detector.release()
            print("\n✅ Stopped\n")
        except Exception as e:
            print(f"❌ Error: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    try:
        print("\n" + "="*60)
        print("VISION ASSISTANT v1.0.0")
        print("="*60 + "\n")
        app = VisionAssistantApp()
        app.run()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal: {e}")
        traceback.print_exc()
        sys.exit(1)
