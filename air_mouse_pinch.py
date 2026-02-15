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
    
    # --- PINCH SETTINGS (The "Pin & Freeze") ---
    # Hand Scale Multipliers:
    # 1. FREEZE_ZONE: When thumb is 15% away from index, STOP cursor.
    FREEZE_THRESHOLD = 0.15 
    # 2. CLICK_ZONE: When thumb is 8% away, CLICK.
    CLICK_THRESHOLD = 0.08
    # 3. RELEASE_ZONE: When thumb is 20% away, RELEASE.
    RELEASE_THRESHOLD = 0.20
    
    # --- DRAG PHYSICS ---
    # How far you must move your hand to "Break" the freeze and start dragging.
    # Prevents accidental drags when just trying to double click.
    DRAG_BREAK_THRESHOLD = 15.0 
    
    # --- MOVEMENT PHYSICS ---
    MOVEMENT_DEADZONE = 4.0 
    SMOOTHING = 3.0       
    FRAME_MARGIN = 100 
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
        
        # Position Sync
        current_x, current_y = pyautogui.position()
        self.raw_x, self.raw_y = current_x, current_y 
        self.curr_x, self.curr_y = current_x, current_y 
        self.drag_start_x, self.drag_start_y = 0, 0
        
        # State Flags
        self.is_pointing = False
        self.is_frozen = False   # NEW: Is cursor locked?
        self.is_clicking = False
        self.click_active = False 
        self.running = True
        
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
        wrist = landmarks[0]
        pip = landmarks[6] 
        tip = landmarks[8] 
        return math.hypot(tip.x - wrist.x, tip.y - wrist.y) > math.hypot(pip.x - wrist.x, pip.y - wrist.y)

    def update_pinch_state(self, landmarks):
        """
        THE PINCH LOGIC
        Returns: 
        1. clicking (bool): Is the click active?
        2. frozen (bool): Should the cursor be frozen?
        """
        wrist = landmarks[0]
        index_mcp = landmarks[5] 
        hand_scale = self.get_distance(wrist, index_mcp)
        
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        pinch_dist = self.get_distance(thumb_tip, index_tip)
        
        # Calculate Thresholds based on hand size
        freeze_lim = hand_scale * Config.FREEZE_THRESHOLD
        click_lim = hand_scale * Config.CLICK_THRESHOLD
        release_lim = hand_scale * Config.RELEASE_THRESHOLD
        
        # 1. Check Freeze (Are we approaching a click?)
        # If we are closer than freeze limit, but not yet clicked -> FREEZE
        should_freeze = (pinch_dist < freeze_lim)
        
        # 2. Check Click (Hysteresis)
        if not self.click_active: 
            if pinch_dist < click_lim:
                self.click_active = True 
        else: 
            if pinch_dist > release_lim:
                self.click_active = False 
        
        return self.click_active, should_freeze, thumb_tip, index_tip

    def vision_worker(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAM_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, Config.FPS)
        
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
                
                pointing = self.is_index_pointing(hand_lms)
                clicking, frozen, t_tip, i_tip = self.update_pinch_state(hand_lms)
                
                self.is_pointing = pointing
                self.is_clicking = clicking
                self.is_frozen = frozen # Pass freeze state to mouse thread
                
                if pointing:
                    target_x = np.interp(idx_tip.x * Config.CAM_WIDTH, 
                                       (Config.FRAME_MARGIN, Config.CAM_WIDTH - Config.FRAME_MARGIN), 
                                       (0, screen_w))
                    target_y = np.interp(idx_tip.y * Config.CAM_HEIGHT, 
                                       (Config.FRAME_MARGIN, Config.CAM_HEIGHT - Config.FRAME_MARGIN), 
                                       (0, screen_h))
                    
                    self.raw_x, self.raw_y = target_x, target_y
                    
                    # VISUAL FEEDBACK
                    if clicking:
                        color = (0, 0, 255) # Red = Click
                        cv2.line(frame, 
                                (int(t_tip.x * Config.CAM_WIDTH), int(t_tip.y * Config.CAM_HEIGHT)),
                                (int(i_tip.x * Config.CAM_WIDTH), int(i_tip.y * Config.CAM_HEIGHT)),
                                color, 3)
                    elif frozen:
                        color = (0, 255, 255) # Yellow = Frozen (Approaching)
                        cv2.circle(frame, (int(idx_tip.x * Config.CAM_WIDTH), int(idx_tip.y * Config.CAM_HEIGHT)), 
                                  10, color, 2)
                    else:
                        color = (0, 255, 0) # Green = Moving
                        cv2.circle(frame, (int(idx_tip.x * Config.CAM_WIDTH), int(idx_tip.y * Config.CAM_HEIGHT)), 
                                  5, color, -1)

            else:
                self.is_pointing = False
            
            status = "FROZEN" if self.is_frozen else "ACTIVE"
            cv2.putText(frame, f"State: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Pin & Freeze Mouse", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break
                
        cap.release()
        cv2.destroyAllWindows()

    def mouse_worker(self):
        print("--- MOUSE THREAD ACTIVE ---")
        
        while self.running:
            if self.is_pointing:
                
                # --- 1. MOVEMENT LOGIC ---
                # Calculate potential new position
                dist_x = self.raw_x - self.curr_x
                dist_y = self.raw_y - self.curr_y
                dist_total = math.hypot(dist_x, dist_y)
                
                # Check Drag Breakout
                # If we are clicking, we only move if we pull HARD (break the freeze)
                is_drag_intention = False
                if self.dragging:
                    drag_dist = math.hypot(self.raw_x - self.drag_start_x, self.raw_y - self.drag_start_y)
                    if drag_dist > Config.DRAG_BREAK_THRESHOLD:
                        is_drag_intention = True

                # UPDATE POSITION IF:
                # 1. Not Frozen (Normal Hover)
                # 2. OR Dragging (Intentionally moving while clicked)
                # 3. AND movement is outside deadzone
                if (not self.is_frozen or is_drag_intention) and dist_total > Config.MOVEMENT_DEADZONE:
                    
                    self.curr_x += dist_x / Config.SMOOTHING
                    self.curr_y += dist_y / Config.SMOOTHING
                    pyautogui.moveTo(self.curr_x, self.curr_y)

                # --- 2. CLICK LOGIC ---
                if self.is_clicking:
                    if not self.dragging:
                        # START CLICK
                        # Save position where click started (for drag detection)
                        self.drag_start_x, self.drag_start_y = self.curr_x, self.curr_y
                        
                        pyautogui.mouseDown()
                        self.dragging = True
                        
                        if time.time() - self.last_click_time < Config.DOUBLE_CLICK_TIME:
                            pyautogui.rightClick()
                        self.last_click_time = time.time()
                else:
                    # END CLICK
                    if self.dragging:
                        pyautogui.mouseUp()
                        self.dragging = False
            
            else:
                # Sync when paused
                real_x, real_y = pyautogui.position()
                self.curr_x, self.curr_y = real_x, real_y
                if self.dragging:
                    pyautogui.mouseUp()
                    self.dragging = False
            
            time.sleep(0.001) 

def main():
    print("--- PIN & FREEZE MOUSE STARTED ---")
    print("1. Point to Move")
    print("2. Pinch to Click (Cursor freezes automatically!)")
    
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