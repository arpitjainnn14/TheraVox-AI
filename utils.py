import os
import cv2
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def create_directories():
    """Create necessary directories if they don't exist."""
    directories = ['logs', 'screenshots']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

def save_screenshot(frame, emotion=None):
    """Save a screenshot with timestamp and emotion label."""
    try:
        # Ensure screenshots directory exists
        screenshots_dir = 'screenshots'
        if not os.path.exists(screenshots_dir):
            os.makedirs(screenshots_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}"
        if emotion:
            filename += f"_{emotion}"
        filename += ".jpg"
        
        filepath = os.path.join(screenshots_dir, filename)
        
        # Save the image
        success = cv2.imwrite(filepath, frame)
        if not success:
            raise Exception("cv2.imwrite failed - unable to save image")
            
        return filepath
        
    except Exception as e:
        print(f"Error saving screenshot: {str(e)}")
        return None


def draw_emotion_box(frame, face_location, emotion, confidence):
    """Draw bounding box and emotion label on frame."""
    x, y, w, h = face_location
    # Draw rectangle around face
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Prepare emotion text
    text = f"{emotion}: {confidence:.2f}"
    
    # Draw background rectangle for text
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    cv2.rectangle(frame, (x, y - 30), (x + text_size[0], y), (0, 255, 0), -1)
    
    # Draw text
    cv2.putText(frame, text, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    return frame

def preprocess_face(face_img, target_size=(48, 48)):
    """Preprocess face image for emotion detection."""
    # Convert to grayscale
    if len(face_img.shape) == 3:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    
    # Resize
    face_img = cv2.resize(face_img, target_size)
    
    # Normalize
    face_img = face_img.astype('float32') / 255.0
    
    # Reshape for model input
    face_img = np.expand_dims(face_img, axis=0)
    face_img = np.expand_dims(face_img, axis=-1)
    
    return face_img 