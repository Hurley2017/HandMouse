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
import threading

# --- CONFIGURATION ---
class Config:
    CAM_WIDTH, CAM_HEIGHT = 640, 480
    FPS = 60 
    
    # --- SENSITIVITY SETTINGS ---
    # Lowered for smaller fingers / subtle clicks
    # 0.12 means the gap needs to be 12% of your hand size to trigger
    CLICK_SENSITIVITY = 0.12 
    RELEASE_SENSITIVITY = 0.30
    
    # --- PHYSICS SETTINGS ---
    # MOVEMENT_DEADZONE: 
    # If the target moves less than 3 pixels, IGNORE IT.
    # This kills the "shaking" when holding still.
    MOVEMENT_DEADZONE = 3.0 
    
    # ADAPTIVE SMOOTHING:
    # Slow movement = High smoothing (Precision)
    # Fast movement = Low smoothing (Snappy)
    MIN_SMOOTHING = 2.0  # For flick shots
    MAX_SMOOTHING = 15.0 # For clicking tiny icons

    FRAME_MARGIN = 80
    DOUBLE_CLICK_TIME = 0.25

pyautogui.FAILSAFE = False 
pyautogui.PAUSE = 0 
screen_w, screen_h = pyautogui.size()

class HandController:
    def __init__(self):
        self.model_path = self._ensure_model_downloaded()
        
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=vision.RunningMode.VIDEO
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        
        # Sync Start Position
        current_x, current_y = pyautogui.position()
        self.raw_x, self.raw_y = current_x, current_y 
        self.curr_x, self.curr_y = current_x, current_y 
        
        self.is_pointing = False
        self.is_pinching = False
        self.running = True
        self.pinch_active = False 
        
        self.dragging = False
        self.last_click_time = 0

    def _ensure_model_downloaded(self):
        model_path = "hand_landmarker.task"
        if not os.path.exists(model_path):
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

    def update_pinch_state(self, landmarks):
        wrist = landmarks[0]
        index_mcp = landmarks[5]
        hand_size = self.get_distance(wrist, index_mcp)
        
        # DYNAMIC THRESHOLDS (Based on hand size)
        click_thresh = hand_size * Config.CLICK_SENSITIVITY
        release_thresh = hand_size * Config.RELEASE_SENSITIVITY
        
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        dist = self.get_distance(thumb_tip, index_tip)
        
        if not self.pinch_active:
            if dist < click_thresh:
                self.pinch_active = True 
        else:
            if dist > release_thresh:
                self.pinch_active = False 
                
        return self.pinch_active, thumb_tip, index_tip

    def vision_worker(self):
        cap = cv2.VideoCapture(0)
        cap.set(3, Config.CAM_WIDTH)
        cap.set(4, Config.CAM_HEIGHT)
        cap.set(5, Config.FPS)
        
        start_time = time.time()
        print("--- VISION THREAD ACTIVE ---")
        
        while self.running:
            success, frame = cap.read()
            if not success: continue
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            timestamp = int((time.time() - start_time) * 1000)
            result = self.landmarker.detect_for_video(mp_image, timestamp)
            
            if result.hand_landmarks:
                hand_lms = result.hand_landmarks[0]
                idx_tip = hand_lms[8]
                
                pointing = not self.is_finger_curled(hand_lms, 8, 6, 0)
                pinching, t_tip, i_tip = self.update_pinch_state(hand_lms)
                
                self.is_pointing = pointing
                self.is_pinching = pinching
                
                if pointing:
                    target_x = np.interp(idx_tip.x * Config.CAM_WIDTH, 
                                       (Config.FRAME_MARGIN, Config.CAM_WIDTH - Config.FRAME_MARGIN), 
                                       (0, screen_w))
                    target_y = np.interp(idx_tip.y * Config.CAM_HEIGHT, 
                                       (Config.FRAME_MARGIN, Config.CAM_HEIGHT - Config.FRAME_MARGIN), 
                                       (0, screen_h))
                    self.raw_x, self.raw_y = target_x, target_y
                
                # Visual Feedback
                if pinching:
                    cv2.line(frame, 
                            (int(t_tip.x * Config.CAM_WIDTH), int(t_tip.y * Config.CAM_HEIGHT)),
                            (int(i_tip.x * Config.CAM_WIDTH), int(i_tip.y * Config.CAM_HEIGHT)),
                            (0, 255, 255), 3)

            else:
                self.is_pointing = False
            
            cv2.imshow("Anti-Jitter Mouse", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break
                
        cap.release()
        cv2.destroyAllWindows()

    def mouse_worker(self):
        print("--- MOUSE THREAD ACTIVE ---")
        
        while self.running:
            if self.is_pointing:
                # --- 1. ADAPTIVE PHYSICS ENGINE ---
                
                # Calculate Distance to Target (Error)
                dist_x = self.raw_x - self.curr_x
                dist_y = self.raw_y - self.curr_y
                dist_total = math.hypot(dist_x, dist_y)
                
                # DEADZONE CHECK:
                # If the AI wants to move < 3 pixels, we assume it's camera noise.
                # We simply DON'T update current position. Mouse stays frozen.
                if dist_total > Config.MOVEMENT_DEADZONE:
                    
                    # DYNAMIC SMOOTHING:
                    # If moving fast (large distance) -> Low Smoothing (Fast)
                    # If moving slow (small distance) -> High Smoothing (Precise)
                    # We map distance (0 to 200px) to smoothing (15.0 to 2.0)
                    
                    # Inverse relationship: More distance = Less smoothing
                    smooth_factor = np.interp(dist_total, (0, 200), (Config.MAX_SMOOTHING, Config.MIN_SMOOTHING))
                    
                    self.curr_x += dist_x / smooth_factor
                    self.curr_y += dist_y / smooth_factor
                    
                    pyautogui.moveTo(self.curr_x, self.curr_y)
                
                # --- 2. CLICK LOGIC ---
                if self.is_pinching:
                    if not self.dragging:
                        pyautogui.mouseDown()
                        self.dragging = True
                        if time.time() - self.last_click_time < Config.DOUBLE_CLICK_TIME:
                            pyautogui.rightClick()
                        self.last_click_time = time.time()
                else:
                    if self.dragging:
                        pyautogui.mouseUp()
                        self.dragging = False
            
            else:
                # Clutch Sync
                real_x, real_y = pyautogui.position()
                self.curr_x, self.curr_y = real_x, real_y
                
                if self.dragging:
                    pyautogui.mouseUp()
                    self.dragging = False
            
            time.sleep(0.001) 

def main():
    controller = HandController()
    t1 = threading.Thread(target=controller.vision_worker)
    t2 = threading.Thread(target=controller.mouse_worker)
    t1.start()
    t2.start()
    t1.join()
    controller.running = False
    t2.join()

if __name__ == "__main__":
    main()