import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import math
import numpy as np
import time
import os
import urllib.request

# --- CONFIGURATION ---
class Config:
    # Native Resolution (Matches Model Training Size = Zero CPU Resize Overhead)
    CAM_WIDTH, CAM_HEIGHT = 640, 480 
    FPS = 60 # Request 60FPS from camera if supported
    
    # Tuning for "Snappiness"
    SMOOTHING = 2.0       # Lowered from 5.0 -> 2.0 for faster response
    PINCH_THRESHOLD = 0.04 
    FRAME_MARGIN = 50     # Smaller margin for 640p
    DOUBLE_CLICK_TIME = 0.25

pyautogui.FAILSAFE = False 
screen_w, screen_h = pyautogui.size()

class HandController:
    def __init__(self):
        self.model_path = self._ensure_model_downloaded()

        # --- THE GPU SWITCH ---
        # We explicitly request the GPU Delegate
        base_options = python.BaseOptions(
            model_asset_path=self.model_path
        )
        
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5, # Lowered slightly for speed
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=vision.RunningMode.VIDEO # OPTIMIZED MODE
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        
        self.prev_x, self.prev_y = 0, 0
        self.dragging = False
        self.last_click_time = 0
        self.start_time = time.time() # For timestamping

    def _ensure_model_downloaded(self):
        model_path = "hand_landmarker.task"
        if not os.path.exists(model_path):
            print("Downloading AI Model...")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
        return model_path

    def get_distance(self, p1, p2):
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def is_finger_curled(self, landmarks, tip_idx, pip_idx, wrist_idx):
        wrist = landmarks[wrist_idx]
        pip = landmarks[pip_idx]
        tip = landmarks[tip_idx]
        return math.hypot(tip.x - wrist.x, tip.y - wrist.y) < math.hypot(pip.x - wrist.x, pip.y - wrist.y)

    def process_frame(self, frame, timestamp_ms):
        # NO resizing. NO flipping (flip at end). Raw speed.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Async detection is complex in Python, so we use VIDEO mode which is faster than IMAGE mode
        detection_result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        
        h, w, _ = frame.shape
        status_text = "Status: Searching..."
        color = (0, 0, 255)

        # Flip for display/logic mapping (doing it once here is faster)
        # However, for mouse mapping, we need to invert X if we don't flip the image
        # Let's flip the coordinates instead of the image to save CPU cycles.
        
        if detection_result.hand_landmarks:
            hand_lms = detection_result.hand_landmarks[0]
            idx_tip = hand_lms[8]
            thumb_tip = hand_lms[4]

            # Logic Check
            is_pointing = not self.is_finger_curled(hand_lms, 8, 6, 0)
            pinch_dist = self.get_distance(idx_tip, thumb_tip)
            is_pinching = pinch_dist < Config.PINCH_THRESHOLD

            if is_pointing:
                # Coordinate Mapping (Inverted X for "Mirror" feel without flipping image)
                norm_x = 1.0 - idx_tip.x 
                norm_y = idx_tip.y
                
                target_x = np.interp(norm_x * w, (Config.FRAME_MARGIN, w - Config.FRAME_MARGIN), (0, screen_w))
                target_y = np.interp(norm_y * h, (Config.FRAME_MARGIN, h - Config.FRAME_MARGIN), (0, screen_h))
                
                # Low Smoothing for Snappiness
                curr_x = self.prev_x + (target_x - self.prev_x) / Config.SMOOTHING
                curr_y = self.prev_y + (target_y - self.prev_y) / Config.SMOOTHING
                
                pyautogui.moveTo(curr_x, curr_y)
                self.prev_x, self.prev_y = curr_x, curr_y
                
                status_text = "HOVER"
                color = (0, 255, 0)

                if is_pinching:
                    if not self.dragging:
                        pyautogui.mouseDown()
                        self.dragging = True
                        if time.time() - self.last_click_time < Config.DOUBLE_CLICK_TIME:
                            pyautogui.rightClick()
                        self.last_click_time = time.time()
                    status_text = "CLICK"
                    color = (0, 255, 255)
                else:
                    if self.dragging:
                        pyautogui.mouseUp()
                        self.dragging = False
            else:
                status_text = "CLUTCH"
                if self.dragging:
                    pyautogui.mouseUp()
                    self.dragging = False

            # Minimal Visuals (Circles are faster than text)
            cv2.circle(frame, (int(idx_tip.x * w), int(idx_tip.y * h)), 5, color, -1)

        # Only flip for the final display window (Visual only, doesn't affect logic speed)
        return cv2.flip(frame, 1)

def main():
    cap = cv2.VideoCapture(0)
    # Force 60FPS and Low Res for Max Speed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, Config.FPS)
    
    controller = HandController()
    print("--- HIGH PERFORMANCE MODE ---")
    
    start_time = time.time()
    
    while True:
        success, frame = cap.read()
        if not success: break
        
        # Calculate timestamp for VIDEO mode
        frame_timestamp_ms = int((time.time() - start_time) * 1000)
        
        processed_frame = controller.process_frame(frame, frame_timestamp_ms)
        cv2.imshow('Fast Mouse', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()