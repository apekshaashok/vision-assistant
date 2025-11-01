"""
Utility functions for Vision Assistant
"""

import cv2
import numpy as np
from collections import Counter


def draw_bounding_box(frame, box, label, confidence, color=(0, 255, 0), thickness=2):
    """Draw bounding box with label"""
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    label_text = f"{label} {confidence:.2f}"
    (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
    cv2.putText(frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return frame


def detect_door_shapes(frame):
    """Enhanced door detection - MORE SENSITIVE"""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # MORE edge detection methods
    edges1 = cv2.Canny(gray, 10, 50)   # Very sensitive
    edges2 = cv2.Canny(gray, 30, 100)  # Medium
    edges3 = cv2.Canny(gray, 50, 150)  # Less sensitive
    edges = cv2.bitwise_or(edges1, cv2.bitwise_or(edges2, edges3))
    
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=3)  # More dilation
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    door_candidates = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        
        # RELAXED criteria
        if width < 30 or height < 80:  # Smaller minimum
            continue
        if width > w * 0.9 or height > h * 0.95:  # Larger maximum
            continue
        
        aspect_ratio = height / width if width > 0 else 0
        area = width * height
        area_ratio = area / (w * h)
        
        # VERY RELAXED door criteria
        is_tall = aspect_ratio > 1.3  # Was 1.5
        is_reasonable_size = 0.02 < area_ratio < 0.7  # Was 0.03-0.6
        not_too_wide = width < w * 0.6  # Was 0.5
        
        if is_tall and is_reasonable_size and not_too_wide:
            score = 1  # Give all candidates a base score
            if 1.8 < aspect_ratio < 3.5:
                score += 3
            if 0.08 < area_ratio < 0.5:
                score += 2
            door_candidates.append({'bbox': [x, y, x + width, y + height], 'score': score, 'area': area})
    
    door_candidates.sort(key=lambda d: (d['score'], d['area']), reverse=True)
    return [d['bbox'] for d in door_candidates[:3]]  # Return top 3


def generate_description(detections):
    """Generate natural language description"""
    if not detections:
        return "I don't see any objects nearby."
    counts = Counter(detections)
    parts = []
    for obj, count in counts.items():
        if count == 1:
            parts.append(f"one {obj}")
        else:
            parts.append(f"{count} {obj}s")
    if len(parts) == 1:
        return f"I see {parts[0]}."
    elif len(parts) == 2:
        return f"I see {parts[0]} and {parts[1]}."
    else:
        return f"I see {', '.join(parts[:-1])}, and {parts[-1]}."


def get_position_info(bbox, frame_width, frame_height):
    """Determine spatial position of object"""
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    obj_width = x2 - x1
    obj_height = y2 - y1
    left_threshold = frame_width * 0.33
    right_threshold = frame_width * 0.66
    if center_x < left_threshold:
        direction = "on your left"
    elif center_x > right_threshold:
        direction = "on your right"
    else:
        direction = "straight ahead"
    box_area = obj_width * obj_height
    frame_area = frame_width * frame_height
    size_ratio = box_area / frame_area
    if size_ratio > 0.15:
        distance = "very close"
    elif size_ratio > 0.05:
        distance = "close"
    elif size_ratio > 0.02:
        distance = "at medium distance"
    else:
        distance = "far away"
    return direction, distance


def calculate_angle_from_center(bbox, frame_width):
    """Calculate angle from center"""
    x1, y1, x2, y2 = bbox
    obj_center_x = (x1 + x2) / 2
    frame_center = frame_width / 2
    offset = obj_center_x - frame_center
    horizontal_fov = 60
    angle = (offset / frame_width) * horizontal_fov
    return int(angle)


def generate_object_query_response(object_name, detections, frame_width, frame_height):
    """Generate response for object queries"""
    from core.config import NON_DETECTABLE_OBJECTS
    if object_name.lower() in NON_DETECTABLE_OBJECTS:
        return f"Sorry, I cannot detect {object_name}s."
    matches = []
    for label, bbox in detections:
        if object_name.lower() in label.lower() or label.lower() in object_name.lower():
            direction, distance = get_position_info(bbox, frame_width, frame_height)
            angle = calculate_angle_from_center(bbox, frame_width)
            matches.append((label, direction, distance, angle))
    if not matches:
        return f"I don't see any {object_name} nearby."
    if len(matches) == 1:
        label, direction, distance, angle = matches[0]
        if angle > 0:
            angle_desc = f"{angle} degrees to your right"
        elif angle < 0:
            angle_desc = f"{abs(angle)} degrees to your left"
        else:
            angle_desc = "straight ahead"
        return f"Yes, I see a {label} at {angle_desc}, {distance}."
    else:
        responses = []
        for label, direction, distance, angle in matches:
            if angle > 0:
                angle_desc = f"{angle} degrees right"
            elif angle < 0:
                angle_desc = f"{abs(angle)} degrees left"
            else:
                angle_desc = "center"
            responses.append(f"one {label} at {angle_desc}, {distance}")
        return f"I see {len(matches)} {object_name}s: {', and '.join(responses)}."


def generate_spatial_description(detections_with_positions):
    """Generate spatial description"""
    if not detections_with_positions:
        return "I don't see any objects nearby."
    descriptions = []
    for label, direction, distance in detections_with_positions:
        desc = f"{label} {direction}, {distance}"
        descriptions.append(desc)
    if len(descriptions) == 1:
        return f"I see one {descriptions[0]}."
    elif len(descriptions) == 2:
        return f"I see {descriptions[0]}, and {descriptions[1]}."
    else:
        return f"I see {', '.join(descriptions[:-1])}, and {descriptions[-1]}."


def check_command(text, command_list):
    """Check if text contains command"""
    return any(cmd in text.lower() for cmd in command_list)


def format_time(seconds):
    """Format time"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs}s" if minutes > 0 else f"{secs}s"


def get_fps(prev_time, curr_time):
    """Calculate FPS"""
    time_diff = curr_time - prev_time
    return 1.0 / time_diff if time_diff > 0 else 0
