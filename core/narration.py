"""
Narration and Text-to-Speech (TTS) Module for Vision Assistant
"""

import pyttsx3
from core.config import TTS_RATE, TTS_VOLUME

class Narrator:
    def __init__(self, rate=TTS_RATE, volume=TTS_VOLUME):
        print(f"Initializing Narrator: TTS rate={rate}, volume={volume}")
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', rate)
            self.engine.setProperty('volume', volume)
            
            # Test TTS on init
            print("Testing TTS engine...")
            self.engine.say("Vision Assistant narrator ready")
            self.engine.runAndWait()
            print("✅ TTS working")
        except Exception as e:
            print(f"⚠️ TTS initialization warning: {e}")
            self.engine = None

    def narrate(self, text):
        """Speaks out the provided text"""
        print(f"🔊 Speaking: {text}")
        
        if self.engine is None:
            print("⚠️ TTS engine not available, skipping speech")
            return
        
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"⚠️ TTS error: {e}")
            # Reinitialize engine if it crashed
            try:
                self.engine = pyttsx3.init()
                self.engine.say(text)
                self.engine.runAndWait()
            except:
                print("❌ TTS failed even after reinit")

# Test the module
if __name__ == "__main__":
    print("Testing Narrator...")
    narrator = Narrator()
    test_text = "Vision Assistant is working! I see one laptop and two bottles."
    narrator.narrate(test_text)
    print("✅ Narration test complete!")