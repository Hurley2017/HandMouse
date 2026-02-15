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

# --- CONFIGURATION (TUNE THESE TO YOUR VIBE) ---
class Config:
    # Camera Settings (Low Res = High Speed)
    CAM_WIDTH, CAM_HEIGHT = 640, 480
    FPS = 60 
    
    # --- MOVEMENT PHYSICS ---
    # DEADZONE: Ignores movements smaller than 6 pixels.
    # This solves the "Stationary Jitter" completely.
    MOVEMENT_DEADZONE = 10.0 
    
    # SMOOTHING: 
    # Low (2.0) = Snappy (Gaming)
    # High (5.0) = Smooth (Desktop)
    SMOOTHING = 5.0       
    
    # --- CLICK SENSITIVITY (Terminator Mode) ---
    # Trigger Point: How far you must curl middle finger to click.
    # 1.4 means "Curl it halfway down".
    CLICK_THRESHOLD = 1.4
    RELEASE_THRESHOLD = 1.6 # Hysteresis gap to prevent flickering
    
    # Screen Mapping
    FRAME_MARGIN = 100 # Virtual border size
    DOUBLE_CLICK_TIME = 0.9 # Max time between clicks for double click (seconds)

# --- PERFORMANCE HACKS ---
pyautogui.FAILSAFE = False 
pyautogui.PAUSE = 0 # ZERO LAG
screen_w, screen_h = pyautogui.size()

class HandController:
    def __init__(self):
        self.model_path = self._ensure_model_downloaded()
        
        # Initialize MediaPipe (Video Mode)
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
        
        # State Flags
        self.is_pointing = False
        self.is_clicking = False
        self.running = True
        self.click_active = False # Latch state
        
        # Mouse Flags
        self.dragging = False
        self.last_click_time = 0

    def _ensure_model_downloaded(self):
        model_path = "hand_landmarker.task"
        if not os.path.exists(model_path):
            print("Downloading AI Model...")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
        return model_path

    def get_distance(self, p1, p2):
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def is_index_pointing(self, landmarks):
        """
        Check if Index Finger is extended (Pointing).
        If curled, we enter 'Clutch Mode' (Stop moving).
        """
        wrist = landmarks[0]
        pip = landmarks[6] # Knuckle
        tip = landmarks[8] # Tip
        
        # Distance from Wrist to Tip vs Wrist to PIP
        dist_tip = math.hypot(tip.x - wrist.x, tip.y - wrist.y)
        dist_pip = math.hypot(pip.x - wrist.x, pip.y - wrist.y)
        
        # If tip is further than PIP, it's extended.
        return dist_tip > dist_pip

    def update_click_state(self, landmarks):
        """
        THE TERMINATOR CLICK (Middle Finger Trigger)
        - Move with Index (8).
        - Click by curling Middle (12).
        """
        # 1. Measure Hand Scale (Wrist to Middle Finger Knuckle)
        # This makes it work for any hand size / distance from camera
        wrist = landmarks[0]
        middle_mcp = landmarks[9] 
        hand_scale = self.get_distance(wrist, middle_mcp)
        
        # 2. Measure Trigger (Wrist to Middle Finger Tip)
        middle_tip = landmarks[12]
        trigger_dist = self.get_distance(wrist, middle_tip)
        
        # 3. Dynamic Thresholds
        # Note: When finger is extended, distance is ~2.0x scale
        # When curled to palm, distance is ~0.8x scale
        click_limit = hand_scale * Config.CLICK_THRESHOLD
        release_limit = hand_scale * Config.RELEASE_THRESHOLD
        
        # 4. State Machine (Hysteresis)
        if not self.click_active: 
            # Currently Released -> Check for Click (Curl Down)
            if trigger_dist < click_limit:
                self.click_active = True 
        else: 
            # Currently Clicked -> Check for Release (Extend Up)
            if trigger_dist > release_limit:
                self.click_active = False 
        
        return self.click_active, middle_tip, middle_mcp

    def vision_worker(self):
        """THREAD 1: The Eyes (Runs at Camera FPS)"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAM_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, Config.FPS)
        
        start_time = time.time()
        print("--- VISION THREAD ACTIVE ---")
        
        while self.running:
            success, frame = cap.read()
            if not success: continue
            
            # Flip & Convert
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # AI Detection
            timestamp = int((time.time() - start_time) * 1000)
            result = self.landmarker.detect_for_video(mp_image, timestamp)
            
            if result.hand_landmarks:
                hand_lms = result.hand_landmarks[0]
                idx_tip = hand_lms[8]
                
                # 1. Is Index Pointing? (Move Logic)
                pointing = self.is_index_pointing(hand_lms)
                
                # 2. Is Middle Curled? (Click Logic)
                clicking, trig_tip, trig_base = self.update_click_state(hand_lms)
                
                # Sync Shared State
                self.is_pointing = pointing
                self.is_clicking = clicking
                
                if pointing:
                    # Map to Screen
                    target_x = np.interp(idx_tip.x * Config.CAM_WIDTH, 
                                       (Config.FRAME_MARGIN, Config.CAM_WIDTH - Config.FRAME_MARGIN), 
                                       (0, screen_w))
                    target_y = np.interp(idx_tip.y * Config.CAM_HEIGHT, 
                                       (Config.FRAME_MARGIN, Config.CAM_HEIGHT - Config.FRAME_MARGIN), 
                                       (0, screen_h))
                    
                    self.raw_x, self.raw_y = target_x, target_y
                    
                    # --- VISUALS: POINTER ---
                    color = (0, 255, 0) # Green = Active
                    cv2.circle(frame, (int(idx_tip.x * Config.CAM_WIDTH), int(idx_tip.y * Config.CAM_HEIGHT)), 
                              8, color, -1)
                
                # --- VISUALS: CLICKER ---
                if clicking:
                    # Draw RED line on Middle Finger when clicked
                    cv2.line(frame, 
                            (int(trig_tip.x * Config.CAM_WIDTH), int(trig_tip.y * Config.CAM_HEIGHT)),
                            (int(trig_base.x * Config.CAM_WIDTH), int(trig_base.y * Config.CAM_HEIGHT)),
                            (0, 0, 255), 4)
                    cv2.putText(frame, "CLICK!", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            else:
                self.is_pointing = False
            
            # Status Text
            status = "ACTIVE" if self.is_pointing else "PAUSED (Clutch)"
            cv2.putText(frame, f"Mode: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            cv2.imshow("Terminator Mouse", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break
                
        cap.release()
        cv2.destroyAllWindows()

    def mouse_worker(self):
        """THREAD 2: The Hands (Runs at 1000Hz)"""
        print("--- MOUSE THREAD ACTIVE ---")
        
        while self.running:
            if self.is_pointing:
                # --- 1. ADAPTIVE PHYSICS ---
                # Calculate Error (Distance to target)
                dist_x = self.raw_x - self.curr_x
                dist_y = self.raw_y - self.curr_y
                dist_total = math.hypot(dist_x, dist_y)
                
                # DEADZONE: If movement is tiny (< 6px), IGNORE IT.
                # This makes the mouse rock solid when holding still.
                if dist_total > Config.MOVEMENT_DEADZONE:
                    
                    # Apply Smoothing
                    self.curr_x += dist_x / Config.SMOOTHING
                    self.curr_y += dist_y / Config.SMOOTHING
                    
                    pyautogui.moveTo(self.curr_x, self.curr_y)
                
                # --- 2. CLICK LOGIC ---
                if self.is_clicking:
                    if not self.dragging:
                        # Mouse Down
                        pyautogui.mouseDown()
                        self.dragging = True
                        
                        # Double Click Check
                        if time.time() - self.last_click_time < Config.DOUBLE_CLICK_TIME:
                            pyautogui.rightClick()
                        self.last_click_time = time.time()
                else:
                    # Mouse Up
                    if self.dragging:
                        pyautogui.mouseUp()
                        self.dragging = False
            
            else:
                # --- 3. CLUTCH SYNC ---
                # When paused, sync to real mouse position
                real_x, real_y = pyautogui.position()
                self.curr_x, self.curr_y = real_x, real_y
                
                if self.dragging:
                    pyautogui.mouseUp()
                    self.dragging = False
            
            time.sleep(0.001) 

def main():
    print("--- TERMINATOR MOUSE STARTED ---")
    print("1. Index Finger -> Move Cursor")
    print("2. Middle Finger Curl -> Click")
    print("3. Fist -> Stop/Reposition")
    
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