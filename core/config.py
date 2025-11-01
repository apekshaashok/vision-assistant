"""
Configuration settings for Vision Assistant
"""

# Model Configuration
MODEL_PATH = 'models/yolov8n.pt'
CONFIDENCE_THRESHOLD = 0.5

# Camera Configuration
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Text-to-Speech Configuration
TTS_RATE = 180
TTS_VOLUME = 1.0

# Speech Recognition Configuration
VOICE_ENERGY_THRESHOLD = 300
VOICE_TIMEOUT = 3
VOICE_PHRASE_LIMIT = 3

# Voice Commands
DESCRIBE_COMMANDS = ["describe", "what", "see", "scan", "look", "surrounding"]
REPEAT_COMMANDS = ["repeat", "again", "last"]
EXIT_COMMANDS = ["stop", "exit", "quit", "bye", "goodbye"]

# Object Query Commands
OBJECT_QUERY_KEYWORDS = [
    "where is", "find", "locate", "is there", 
    "do you see", "can you see", "show me"
]

# Queryable Objects
QUERYABLE_OBJECTS = [
    "door", "doors", "exit",
    "person", "people", "chair", "couch", "table", "bed",
    "laptop", "computer", "tv", "monitor", "keyboard", "mouse",
    "phone", "cell phone", "bottle", "cup", "bowl",
    "book", "clock", "vase", "bag", "backpack", "suitcase",
    "car", "bicycle", "motorcycle", "bus", "truck",
    "refrigerator", "microwave", "oven", "sink", "toilet"
]

# Non-detectable objects
NON_DETECTABLE_OBJECTS = [
    "window", "wall", "ceiling", "floor", 
    "room", "stairs", "elevator", "hallway"
]

# UI Configuration
WINDOW_NAME = "Vision Assistant"
BOX_COLOR = (0, 255, 0)
BOX_THICKNESS = 2
FONT = 0
FONT_SCALE = 0.6
FONT_COLOR = (0, 255, 0)
FONT_THICKNESS = 2

# Application Info
APP_NAME = "Vision Assistant"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "AI-Powered Accessibility Tool"

# Emotion Detection
EMOTION_DETECTION_ENABLED = True


print(f"✅ {APP_NAME} v{APP_VERSION} - Configuration loaded")
