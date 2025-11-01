import sys

print("Step 1: Importing ObjectDetector...")
from core.detection import ObjectDetector
print("✅ ObjectDetector imported")

print("Step 2: Initializing ObjectDetector...")
detector = ObjectDetector()
print("✅ ObjectDetector initialized")

print("Step 3: Importing Narrator...")
from core.narration import Narrator
print("✅ Narrator imported")

print("Step 4: Initializing Narrator...")
narrator = Narrator()
print("✅ Narrator initialized")

print("Step 5: Importing VoiceController...")
from core.voice_control import VoiceController
print("✅ VoiceController imported")

print("Step 6: Initializing VoiceController...")
voice_ctrl = VoiceController()
print("✅ VoiceController initialized")

print("Step 7: Importing TextReader...")
from core.ocr import TextReader
print("✅ TextReader imported")

print("Step 8: Initializing TextReader...")
text_reader = TextReader()
print("✅ TextReader initialized")

print("\n🎉 All modules initialized successfully!")
