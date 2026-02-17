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
    
    # --- DIRECTION & SPEED ---
    DIR_X = -1 
    DIR_Y = -1
    SPEED_MULTIPLIER = 4.5
    SCROLL_SPEED = 30 # Pixels per scroll step
    
    # --- GESTURE THRESHOLDS ---
    # Hand Scale Multipliers
    PINCH_CLICK_DIST = 0.06   # Gap to trigger click
    SCROLL_MODE_DIST = 0.05   # Gap between Index/Middle to be considered "Together"
    
    # --- PHYSICS ---
    MOVEMENT_DEADZONE = 0.5 
    SMOOTHING = 3.0

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
        
        # State Vectors (Live Axis)
        self.axis_forward = np.array([0.0, -1.0]) 
        self.axis_side = np.array([1.0, 0.0])
        
        # Position State
        self.curr_x_px = 0
        self.curr_y_px = 0
        
        # Gesture Flags
        self.mode = "CLUTCH" # CLUTCH, MOVE, SCROLL
        self.is_clicking_left = False
        self.is_clicking_right = False
        self.running = True

    def _ensure_model_downloaded(self):
        model_path = "hand_landmarker.task"
        if not os.path.exists(model_path):
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
        return model_path

    def get_dist(self, p1, p2):
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def update_live_axis(self, wrist, middle_mcp):
        """Re-calculates 'Up' and 'Right' based on hand orientation"""
        wx, wy = wrist.x * Config.CAM_WIDTH, wrist.y * Config.CAM_HEIGHT
        mx, my = middle_mcp.x * Config.CAM_WIDTH, middle_mcp.y * Config.CAM_HEIGHT
        
        vec_x = mx - wx
        vec_y = my - wy 
        length = math.hypot(vec_x, vec_y)
        if length < 1e-6: return
        
        # Forward Vector
        ux, uy = vec_x / length, vec_y / length
        # Side Vector (Rotated 90 deg)
        vx, vy = -uy, ux
        
        self.axis_forward = np.array([ux, uy])
        self.axis_side = np.array([vx, vy])

    def detect_state(self, lms):
        """
        DETERMINES MODE:
        1. Fist/Curled = CLUTCH (Stop)
        2. Index Extended = MOVE
        3. Index + Middle Extended = SCROLL
        """
        # Finger Tips & PIPs (Knuckles)
        idx_tip = lms[8]; idx_pip = lms[6]
        mid_tip = lms[12]; mid_pip = lms[10]
        ring_tip = lms[16]; ring_pip = lms[14]
        
        wrist = lms[0]
        
        # Check Extensions (Tip further from wrist than PIP)
        idx_ext = self.get_dist(idx_tip, wrist) > self.get_dist(idx_pip, wrist)
        mid_ext = self.get_dist(mid_tip, wrist) > self.get_dist(mid_pip, wrist)
        ring_ext = self.get_dist(ring_tip, wrist) > self.get_dist(ring_pip, wrist)
        
        # LOGIC TREE
        if idx_ext and mid_ext:
            return "SCROLL"
        elif idx_ext and not mid_ext:
            return "MOVE"
        else:
            return "CLUTCH" # Fist or relaxed

    def vision_worker(self):
        cap = cv2.VideoCapture(0)
        cap.set(3, Config.CAM_WIDTH); cap.set(4, Config.CAM_HEIGHT); cap.set(5, Config.FPS)
        start_time = time.time()
        
        while self.running:
            success, frame = cap.read()
            if not success: continue
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp = int((time.time() - start_time) * 1000)
            
            result = self.landmarker.detect_for_video(mp_image, timestamp)
            
            if result.hand_landmarks:
                lms = result.hand_landmarks[0]
                idx_tip = lms[8]
                thumb_tip = lms[4]
                mid_tip = lms[12]
                
                # 1. Update Physics
                self.update_live_axis(lms[0], lms[9])
                self.curr_x_px = idx_tip.x * Config.CAM_WIDTH
                self.curr_y_px = idx_tip.y * Config.CAM_HEIGHT
                
                # 2. Update Mode
                self.mode = self.detect_state(lms)
                
                # 3. Check Clicks (Visual Pinch)
                # Dynamic Threshold
                scale = self.get_dist(lms[0], lms[5])
                thresh = scale * Config.PINCH_CLICK_DIST
                
                # Left Click (Index + Thumb)
                self.is_clicking_left = self.get_dist(idx_tip, thumb_tip) < thresh
                
                # Right Click (Middle + Thumb)
                # Only check if in MOVE mode (to avoid accidental scroll clicks)
                if self.mode == "MOVE":
                    self.is_clicking_right = self.get_dist(mid_tip, thumb_tip) < thresh
                
                # --- VISUALS ---
                cx, cy = int(self.curr_x_px), int(self.curr_y_px)
                
                if self.mode == "MOVE":
                    color = (0, 255, 0) # Green
                    # Draw Arrow
                    end_x = int(cx + self.axis_forward[0] * 40)
                    end_y = int(cy + self.axis_forward[1] * 40)
                    cv2.arrowedLine(frame, (cx, cy), (end_x, end_y), color, 2)
                    
                elif self.mode == "SCROLL":
                    color = (255, 0, 255) # Magenta
                    cv2.putText(frame, "SCROLL MODE", (cx, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                else:
                    color = (0, 0, 255) # Red (Clutch)
                
                if self.is_clicking_left: color = (0, 255, 255) # Yellow Click
                
                cv2.circle(frame, (cx, cy), 8, color, -1)
                
            # Status Text
            cv2.putText(frame, f"MODE: {self.mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow("Virtual Trackpad", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): self.running = False; break
                
        cap.release()
        cv2.destroyAllWindows()

    def mouse_worker(self):
        time.sleep(1.0)
        prev_x = self.curr_x_px
        prev_y = self.curr_y_px
        is_dragging = False
        
        while self.running:
            curr_x = self.curr_x_px
            curr_y = self.curr_y_px
            
            # --- DELTA CALCULATION ---
            raw_dx = curr_x - prev_x
            raw_dy = curr_y - prev_y
            dist = math.hypot(raw_dx, raw_dy)
            
            if dist > Config.MOVEMENT_DEADZONE:
                # Project onto Live Axis
                # Invert forward (Screen Y is down)
                forward_component = -(raw_dx * self.axis_forward[0] + raw_dy * self.axis_forward[1])
                side_component = (raw_dx * self.axis_side[0] + raw_dy * self.axis_side[1])
                
                # --- APPLY BASED ON MODE ---
                
                if self.mode == "MOVE":
                    # Mouse Movement
                    move_x = side_component * Config.SPEED_MULTIPLIER * Config.DIR_X
                    move_y = -forward_component * Config.SPEED_MULTIPLIER * Config.DIR_Y
                    pyautogui.moveRel(move_x, move_y, _pause=False)
                    
                elif self.mode == "SCROLL":
                    # Scroll Wheel
                    # We use the Forward Component to determine scroll direction
                    scroll_amount = int(forward_component * Config.SCROLL_SPEED)
                    if abs(scroll_amount) > 0:
                        pyautogui.scroll(scroll_amount)
            
            # Update history (This creates the "Ratchet" effect)
            # We ALWAYS update prev, so when you declutch and move, it doesn't jump.
            prev_x = curr_x
            prev_y = curr_y

            # --- CLICK LOGIC ---
            if self.is_clicking_left:
                if not is_dragging:
                    pyautogui.mouseDown()
                    is_dragging = True
            else:
                if is_dragging:
                    pyautogui.mouseUp()
                    is_dragging = False
                    
            if self.is_clicking_right and not is_dragging:
                 pyautogui.rightClick()
                 time.sleep(0.3)
            
            time.sleep(0.01)

def main():
    print("--- VIRTUAL TRACKPAD ---")
    print("1. Index Finger -> Move")
    print("2. Two Fingers -> Scroll")
    print("3. Fist/Curl -> Clutch (Reposition)")
    print("4. Pinch -> Click")
    
    controller = HandController()
    t1 = threading.Thread(target=controller.vision_worker)
    t2 = threading.Thread(target=controller.mouse_worker)
    t1.start(); t2.start(); t1.join(); controller.running = False; t2.join()

if __name__ == "__main__":
    main()