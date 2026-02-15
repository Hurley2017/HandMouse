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
import sounddevice as sd

# --- CONFIGURATION (TUNE THESE!) ---
class Config:
    CAM_WIDTH, CAM_HEIGHT = 640, 480
    FPS = 60 
    
    # --- DIRECTION CORRECTION ---
    # If mouse moves opposite to hand, change 1 to -1
    DIR_X = -1  # Try 1 or -1
    DIR_Y = -1  # Try 1 or -1
    
    # --- SENSITIVITY ---
    # How fast the mouse moves relative to hand speed
    SPEED_MULTIPLIER = 3.0
    
    # --- AUDIO THRESHOLDS (CHECK CONSOLE FOR VALUES) ---
    SLIDE_THRESHOLD = 0.5  # Hiss volume
    TAP_THRESHOLD = 8.0    # Thud volume
    
    # --- PHYSICS ---
    MOVEMENT_DEADZONE = 0.002 # Normalized distance (very small)
    SMOOTHING = 5.0       
    TAP_COOLDOWN = 0.4

pyautogui.FAILSAFE = False 
pyautogui.PAUSE = 0 
screen_w, screen_h = pyautogui.size()

class AudioEngine:
    def __init__(self):
        self.running = True
        self.is_sliding = False
        self.is_tapping = False
        self.current_volume = 0.0
        self.stream = None

    def audio_callback(self, indata, frames, time, status):
        # Calculate Volume (Simple Norm)
        vol = np.linalg.norm(indata) * 10
        self.current_volume = vol
        
        # DEBUG: Uncomment to see volume in console live
        # print(f"VOL: {vol:.2f}") 
        
        # LOGIC
        if vol > Config.TAP_THRESHOLD:
            self.is_tapping = True
            self.is_sliding = True
        elif vol > Config.SLIDE_THRESHOLD:
            self.is_sliding = True
            self.is_tapping = False
        else:
            self.is_sliding = False
            self.is_tapping = False

    def start(self):
        try:
            self.stream = sd.InputStream(
                callback=self.audio_callback,
                channels=1,
                samplerate=44100,
                blocksize=1024
            )
            self.stream.start()
            print("--- MIC LISTENING ---")
        except Exception as e:
            print(f"MIC ERROR: {e}")

    def stop(self):
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()

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
        
        # Audio
        self.audio = AudioEngine()
        self.audio.start()
        
        # Relative Tracking State
        self.prev_hand_x = 0
        self.prev_hand_y = 0
        
        # Mouse State
        curr_x, curr_y = pyautogui.position()
        self.mouse_x, self.mouse_y = curr_x, curr_y
        
        self.running = True

    def _ensure_model_downloaded(self):
        model_path = "hand_landmarker.task"
        if not os.path.exists(model_path):
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
        return model_path

    def vision_worker(self):
        cap = cv2.VideoCapture(0)
        cap.set(3, Config.CAM_WIDTH)
        cap.set(4, Config.CAM_HEIGHT)
        cap.set(5, Config.FPS)
        
        start_time = time.time()
        
        while self.running:
            success, frame = cap.read()
            if not success: continue
            
            # 1. Flip frame for user comfort (Mirror)
            frame = cv2.flip(frame, 1)
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp = int((time.time() - start_time) * 1000)
            
            result = self.landmarker.detect_for_video(mp_image, timestamp)
            
            if result.hand_landmarks:
                # Raw Normalized Coordinates (0.0 to 1.0)
                idx_tip = result.hand_landmarks[0][8]
                
                # We simply store the raw hand position here.
                # The logic happens in mouse_worker to ensure smooth deltas.
                self.target_hand_x = idx_tip.x
                self.target_hand_y = idx_tip.y
                
                # Visuals
                color = (0, 255, 0) if self.audio.is_sliding else (0, 0, 255)
                if self.audio.is_tapping: color = (0, 255, 255)
                
                cx, cy = int(idx_tip.x * Config.CAM_WIDTH), int(idx_tip.y * Config.CAM_HEIGHT)
                cv2.circle(frame, (cx, cy), 8, color, -1)
            
            # HUD
            vol = self.audio.current_volume
            # Dynamic Bar color
            bar_col = (0, 255, 0)
            if vol > Config.TAP_THRESHOLD: bar_col = (0, 0, 255)
            elif vol > Config.SLIDE_THRESHOLD: bar_col = (255, 255, 0)
            
            cv2.rectangle(frame, (50, 400), (50 + int(vol * 20), 420), bar_col, -1)
            cv2.putText(frame, f"Vol: {vol:.2f}", (50, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow("Delta Mouse", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break
                
        cap.release()
        cv2.destroyAllWindows()
        self.audio.stop()

    def mouse_worker(self):
        last_tap = 0
        
        # Initialize Previous Hand Position
        # We need a brief moment to get the first frame data
        time.sleep(1) 
        try:
            prev_hx = self.target_hand_x
            prev_hy = self.target_hand_y
        except:
            prev_hx, prev_hy = 0.5, 0.5
            
        while self.running:
            try:
                curr_hx = self.target_hand_x
                curr_hy = self.target_hand_y
            except:
                continue # No hand detected yet

            # 1. MOVEMENT (RELATIVE DELTA)
            if self.audio.is_sliding and not self.audio.is_tapping:
                # Calculate Delta (How much did hand move?)
                dx = curr_hx - prev_hx
                dy = curr_hy - prev_hy
                
                # Apply Deadzone (Ignore tiny jitter)
                if abs(dx) > Config.MOVEMENT_DEADZONE or abs(dy) > Config.MOVEMENT_DEADZONE:
                    
                    # Apply Speed & Direction
                    # Map normalized delta to screen pixels
                    move_x = dx * screen_w * Config.SPEED_MULTIPLIER * Config.DIR_X
                    move_y = dy * screen_h * Config.SPEED_MULTIPLIER * Config.DIR_Y
                    
                    # Move Relative to CURRENT mouse position
                    pyautogui.moveRel(move_x, move_y, _pause=False)
                
                # Update "Previous" to "Current" so we don't drift
                prev_hx = curr_hx
                prev_hy = curr_hy
                
            else:
                # CLUTCH MODE (Silence)
                # We update prev_hx/hy continuously even when not moving mouse.
                # This ensures that when you start moving again, 
                # the delta is calculated from where your hand IS, not where it WAS.
                # This fixes the "Snap Back" glitch.
                prev_hx = curr_hx
                prev_hy = curr_hy
                
                # Debug print for Thud tuning
                if self.audio.current_volume > 1.0:
                    print(f"LOUD NOISE: {self.audio.current_volume:.2f}")

            # 2. CLICK
            if self.audio.is_tapping:
                if time.time() - last_tap > Config.TAP_COOLDOWN:
                    print(f"CLICK! (Vol: {self.audio.current_volume:.2f})")
                    pyautogui.click()
                    last_tap = time.time()
            
            time.sleep(0.01) # 100Hz is enough for relative

def main():
    print("--- DELTA MOUSE STARTED ---")
    print("1. Hiss/Slide -> Moves relative to current position")
    print("2. Thud -> Click")
    print("3. Check Console for Volume numbers!")
    
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