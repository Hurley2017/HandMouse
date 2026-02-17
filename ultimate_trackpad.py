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

# --- CORE ENGINE CONFIGURATION ---
class Config:
    CAM_WIDTH, CAM_HEIGHT = 640, 480
    FPS = 60 
    
    # Movement Tuning
    SPEED = 5.0            # Overall sensitivity
    SMOOTHING = 4.0        # Higher = more "weight" / less jitter
    DEADZONE = 1.2         # Pixels to ignore (kills stationary jitter)
    SCROLL_SENSE = 15.0    # Scrolling speed
    
    # Gesture Tuning
    L_CLICK_THRESH = 0.30  # Thumb-to-knuckle distance ratio
    R_CLICK_COOLDOWN = 0.4 # Seconds between right clicks
    
    # Coordinate Logic
    DIR_X = -1 
    DIR_Y = -1

# Initialize PyAutoGUI Safety
pyautogui.FAILSAFE = False 
pyautogui.PAUSE = 0 
screen_w, screen_h = pyautogui.size()

class NeuralTrackpad:
    def __init__(self):
        self.model_path = self._prep_model()
        
        # MediaPipe Setup
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
        
        # Motion Vectors
        self.axis_fwd = np.array([0.0, -1.0]) 
        self.axis_side = np.array([1.0, 0.0])
        self.hx, self.hy = 0.5, 0.5 # Current Hand Normalized
        
        # State Machine
        self.mode = "CLUTCH" 
        self.l_down = False
        self.r_req = False
        self.running = True

    def _prep_model(self):
        path = "hand_landmarker.task"
        if not os.path.exists(path):
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
            urllib.request.urlretrieve(url, path)
        return path

    def _get_dist(self, p1, p2):
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def _update_orientation(self, wrist, mcp):
        """Calculates the hand's local coordinate system."""
        vx, vy = (mcp.x - wrist.x) * Config.CAM_WIDTH, (mcp.y - wrist.y) * Config.CAM_HEIGHT
        mag = math.hypot(vx, vy)
        if mag < 1e-6: return
        self.axis_fwd = np.array([vx/mag, vy/mag])
        self.axis_side = np.array([-self.axis_fwd[1], self.axis_fwd[0]])

    def _analyze_gestures(self, lms):
        wrist = lms[0]
        
        # Finger extensions
        idx_up = self._get_dist(lms[8], wrist) > self._get_dist(lms[6], wrist)
        mid_up = self._get_dist(lms[12], wrist) > self._get_dist(lms[10], wrist)
        
        # Mode Logic
        if idx_up and mid_up: m = "SCROLL"
        elif idx_up: m = "MOVE"
        else: m = "CLUTCH"
        
        # Click Logic
        scale = self._get_dist(lms[0], lms[5])
        l_click = self._get_dist(lms[4], lms[5]) < (scale * Config.L_CLICK_THRESH)
        r_click = (not mid_up) and (m == "MOVE")
        
        return m, l_click, r_click

    def vision_thread(self):
        cap = cv2.VideoCapture(0)
        cap.set(3, Config.CAM_WIDTH); cap.set(4, Config.CAM_HEIGHT)
        t_start = time.time()
        
        while self.running:
            ret, frame = cap.read()
            if not ret: continue
            
            frame = cv2.flip(frame, 1)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = self.landmarker.detect_for_video(mp_img, int((time.time() - t_start) * 1000))
            
            if res.hand_landmarks:
                l = res.hand_landmarks[0]
                self._update_orientation(l[0], l[9])
                self.hx, self.hy = l[8].x, l[8].y # Always update for Shadow Tracking
                self.mode, self.l_down, self.r_req = self._analyze_gestures(l)
                
                # Draw UI
                cx, cy = int(self.hx * Config.CAM_WIDTH), int(self.hy * Config.CAM_HEIGHT)
                color = (0, 255, 0) if self.mode == "MOVE" else (0, 0, 255)
                if self.mode == "SCROLL": color = (255, 0, 255)
                cv2.circle(frame, (cx, cy), 10, color, -1)
                
                # Forward Arrow
                cv2.arrowedLine(frame, (cx, cy), 
                                (int(cx + self.axis_fwd[0]*40), int(cy + self.axis_fwd[1]*40)), (255,0,0), 2)
                
            cv2.putText(frame, f"MODE: {self.mode}", (10, 30), 1, 1, (255,255,255), 2)
            cv2.imshow("Neural Trackpad v1", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): self.running = False
        cap.release(); cv2.destroyAllWindows()

    def mouse_thread(self):
        time.sleep(1.0)
        px, py = self.hx, self.hy
        active_l_click = False
        last_r = 0
        
        while self.running:
            # 1. Delta Physics (Continuous)
            dx = (self.hx - px) * Config.CAM_WIDTH
            dy = (self.hy - py) * Config.CAM_HEIGHT
            
            # Project to Local Axis
            fwd_comp = -(dx * self.axis_fwd[0] + dy * self.axis_fwd[1])
            side_comp = (dx * self.axis_side[0] + dy * self.axis_side[1])
            
            # 2. Apply Movement
            mag = math.hypot(fwd_comp, side_comp)
            if mag > Config.DEADZONE:
                if self.mode == "MOVE":
                    rx = side_comp * Config.SPEED * Config.DIR_X / Config.SMOOTHING
                    ry = -fwd_comp * Config.SPEED * Config.DIR_Y / Config.SMOOTHING
                    pyautogui.moveRel(rx, ry, _pause=False)
                elif self.mode == "SCROLL":
                    pyautogui.scroll(int(fwd_comp * Config.SCROLL_SENSE))

            # Update history (Fixes the Reset Glitch)
            px, py = self.hx, self.hy

            # 3. Click Execution
            if self.l_down:
                if not active_l_click:
                    pyautogui.mouseDown(); active_l_click = True
            else:
                if active_l_click:
                    pyautogui.mouseUp(); active_l_click = False
            
            if self.r_req and (time.time() - last_r > Config.R_CLICK_COOLDOWN):
                pyautogui.rightClick()
                last_r = time.time()
            
            time.sleep(0.005)

if __name__ == "__main__":
    nt = NeuralTrackpad()
    threading.Thread(target=nt.vision_thread).start()
    threading.Thread(target=nt.mouse_thread).start()