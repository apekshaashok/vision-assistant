"""
OCR (Optical Character Recognition) Module for Vision Assistant
Optimized preprocessing for real-world camera text detection
"""

import ssl
import certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import easyocr
import cv2
import numpy as np


class TextReader:
    def __init__(self, languages=['en']):
        """
        Initialize EasyOCR reader
        Args:
            languages: List of language codes (e.g., ['en'] for English, ['en', 'hi'] for English+Hindi)
        """
        print(f"Initializing OCR reader for languages: {languages}")
        self.reader = easyocr.Reader(languages, gpu=False)  # Set gpu=True if you have CUDA
        print("✅ OCR reader initialized")
    
    def preprocess_image(self, frame):
        """
        Gentle preprocessing that doesn't distort text
        Args:
            frame: Input image
        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply very gentle denoising (less aggressive than before)
        # Using fastNlMeansDenoising instead of bilateral for better text preservation
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        
        # Gentle contrast enhancement using CLAHE with lower clip limit
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Sharpen the image to make text edges crisp
        kernel_sharpen = np.array([[-1,-1,-1],
                                   [-1, 9,-1],
                                   [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel_sharpen)
        
        return sharpened
    
    def read_text(self, frame, confidence_threshold=0.3, preprocess=True):
        """
        Detect and extract text from frame
        Args:
            frame: Image frame
            confidence_threshold: Minimum confidence (lowered for more detections)
            preprocess: Whether to apply image preprocessing
        Returns:
            Tuple of (detected_texts list, annotated_frame)
        """
        # Store original frame for annotation
        original_frame = frame.copy()
        
        # Try BOTH preprocessing and raw - use results from whichever gives more confident results
        results_raw = self.reader.readtext(frame)
        
        if preprocess:
            processed_frame = self.preprocess_image(frame)
            # Convert back to BGR for EasyOCR
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
            results_processed = self.reader.readtext(processed_frame)
            
            # Compare average confidence - use better one
            avg_conf_raw = sum(r[2] for r in results_raw) / len(results_raw) if results_raw else 0
            avg_conf_processed = sum(r[2] for r in results_processed) / len(results_processed) if results_processed else 0
            
            # Use whichever has higher average confidence
            results = results_processed if avg_conf_processed > avg_conf_raw else results_raw
            print(f"  [OCR Debug] Raw conf: {avg_conf_raw:.2f}, Processed conf: {avg_conf_processed:.2f}")
        else:
            results = results_raw
        
        detected_texts = []
        for (bbox, text, conf) in results:
            if conf >= confidence_threshold:
                # Clean up detected text
                text = text.strip()
                if text:  # Only add non-empty text
                    detected_texts.append(text)
                    
                    # Convert bbox to proper numpy array format for OpenCV
                    bbox = np.array(bbox, dtype=np.int32).reshape((-1, 1, 2))
                    
                    # Draw bounding box around text on ORIGINAL frame
                    cv2.polylines(original_frame, [bbox], True, (0, 255, 0), 2)  # Green boxes
                    
                    # Add text label with confidence (get top-left corner)
                    top_left = tuple(bbox[0][0])
                    label = f"{text} ({conf:.2f})"
                    
                    # Add background for text label for better visibility
                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(original_frame, 
                                 (top_left[0], top_left[1] - label_h - 5),
                                 (top_left[0] + label_w, top_left[1]),
                                 (0, 255, 0), -1)
                    
                    cv2.putText(original_frame, label, 
                               (top_left[0], top_left[1] - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return detected_texts, original_frame
    
    def format_text_output(self, texts):
        """
        Format detected texts into natural speech output
        Args:
            texts: List of detected text strings
        Returns:
            String for TTS narration
        """
        if not texts:
            return "No text detected."
        
        if len(texts) == 1:
            return f"Text detected: {texts[0]}"
        else:
            # Join multiple texts with commas
            return f"Text detected: {', '.join(texts)}"


# Test the module
if __name__ == "__main__":
    print("Testing Optimized TextReader...")
    
    # Create test image with text
    test_frame = np.zeros((200, 400, 3), dtype=np.uint8)
    cv2.putText(test_frame, "EXIT TEAM", (30, 100), 
               cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    
    reader = TextReader()
    
    print("\n--- Testing without preprocessing ---")
    texts1, annotated1 = reader.read_text(test_frame.copy(), preprocess=False)
    print(f"Detected: {texts1}")
    
    print("\n--- Testing WITH preprocessing ---")
    texts2, annotated2 = reader.read_text(test_frame.copy(), preprocess=True)
    print(f"Detected: {texts2}")
    
    output = reader.format_text_output(texts2)
    print(f"Output: {output}")
    
    print("\n✅ Optimized OCR module works!")
