"""
Voice Command Recognition Module for Vision Assistant (OPTIMIZED)
Faster response + Better voice pickup
"""

import speech_recognition as sr

# Optimized defaults if config not available
try:
    from core.config import VOICE_ENERGY_THRESHOLD, VOICE_TIMEOUT, VOICE_PHRASE_LIMIT
except ImportError:
    VOICE_ENERGY_THRESHOLD = 300  # Much lower for better sensitivity
    VOICE_TIMEOUT = 3             # Faster timeout
    VOICE_PHRASE_LIMIT = 3        # Shorter phrases


class VoiceController:
    def __init__(self, energy_threshold=VOICE_ENERGY_THRESHOLD):
        print(f"Initializing VoiceController (threshold={energy_threshold})")
        self.recognizer = sr.Recognizer()
        
        # OPTIMIZATION 1: Lower threshold for better voice pickup
        self.recognizer.energy_threshold = energy_threshold
        
        # OPTIMIZATION 2: Enable dynamic threshold adjustment
        self.recognizer.dynamic_energy_threshold = True
        
        # OPTIMIZATION 3: Reduce pause threshold for faster response
        self.recognizer.pause_threshold = 0.5  # Was 0.8 by default
        
        # OPTIMIZATION 4: Initialize microphone once
        self.microphone = sr.Microphone()
        
        # OPTIMIZATION 5: Calibrate once at startup (not every listen)
        print("Calibrating microphone...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
        print(f"✅ Calibrated (threshold: {self.recognizer.energy_threshold})")

    def listen(self):
        """Listens for a command and returns recognized text, or None if failed"""
        with self.microphone as source:
            print("🎤 Listening...")  # Shorter message
            try:
                # Listen with optimized settings
                audio = self.recognizer.listen(
                    source, 
                    timeout=VOICE_TIMEOUT,           # Faster timeout
                    phrase_time_limit=VOICE_PHRASE_LIMIT  # Shorter phrases
                )
            except sr.WaitTimeoutError:
                return None  # Silent fail for speed
            except Exception as e:
                print(f"⚠️ Mic error: {e}")
                return None

        # Recognize speech
        try:
            text = self.recognizer.recognize_google(audio)
            print(f"✅ '{text}'")  # Shorter output
            return text.lower()
        except sr.UnknownValueError:
            return None  # Silent fail for speed
        except sr.RequestError as e:
            print(f"❌ Network error: {e}")
            return None
        except Exception:
            return None


# Test the module
if __name__ == "__main__":
    print("Testing optimized voice control...")
    vc = VoiceController()
    print("Say something like 'describe' or 'stop':")
    command = vc.listen()
    if command:
        print(f"✅ Command: {command}")
    else:
        print("⚠️ No command detected")
    print("✅ Test complete!")
