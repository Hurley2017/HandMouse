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

# --- CONFIGURATION (HIGH SENSITIVITY) ---
class Config:
    CAM_WIDTH, CAM_HEIGHT = 640, 480
    FPS = 60 
    
    # --- DIRECTION CORRECTION ---
    # Change these to -1 if mouse moves opposite to hand
    DIR_X = 1 
    DIR_Y = 1 
    
    # --- SPEED ---
    SPEED_MULTIPLIER = 5.0 # Increased for snappier movement
    
    # --- AUDIO THRESHOLDS (Aggressive) ---
    # SLIDE: Volume needed to MOVE (Hissing)
    # 0.08 is very sensitive. If cursor drifts, raise to 0.12
    SLIDE_THRESHOLD = 0.08   
    
    # TAP: Volume needed to CLICK (Thud)
    # 1.5 is very low. A light tap is enough.
    TAP_THRESHOLD = 1.5     
    
    # --- PHYSICS ---
    MOVEMENT_DEADZONE = 0.002 
    TAP_COOLDOWN = 0.25 # Reduced cooldown for faster clicking

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
        
        # Dynamic Noise Floor Calibration
        self.background_noise = 0.0
        self.calibration_samples = []

    def audio_callback(self, indata, frames, time, status):
        # Calculate Volume
        vol = np.linalg.norm(indata) * 10
        self.current_volume = vol
        
        # Calibration Phase (First 50 samples)
        if len(self.calibration_samples) < 50:
            self.calibration_samples.append(vol)
            return

        # Calculate Logic
        # We check relative to background noise if calibrated
        floor = np.mean(self.calibration_samples)
        
        # TAP LOGIC (Priority)
        if vol > Config.TAP_THRESHOLD:
            self.is_tapping = True
            self.is_sliding = True # Tap includes contact
        
        # SLIDE LOGIC
        # Must be louder than floor AND louder than threshold
        elif vol > (floor + Config.SLIDE_THRESHOLD):
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
            print("--- CALIBRATING MIC (Please remain silent for 1 second)... ---")
            time.sleep(1.0)
            print("--- READY! ---")
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
        
        # Start Audio
        self.audio = AudioEngine()
        self.audio.start()
        
        # Physics State
        self.target_hand_x = 0.5
        self.target_hand_y = 0.5
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
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            timestamp = int((time.time() - start_time) * 1000)
            result = self.landmarker.detect_for_video(mp_image, timestamp)
            
            if result.hand_landmarks:
                idx_tip = result.hand_landmarks[0][8]
                self.target_hand_x = idx_tip.x
                self.target_hand_y = idx_tip.y
                
                # Visuals
                color = (0, 0, 255) # Red = Idle
                if self.audio.is_sliding: color = (0, 255, 0) # Green = Moving
                if self.audio.is_tapping: color = (0, 255, 255) # Yellow = Click
                
                cx, cy = int(idx_tip.x * Config.CAM_WIDTH), int(idx_tip.y * Config.CAM_HEIGHT)
                cv2.circle(frame, (cx, cy), 10, color, -1)
            
            # HUD (Sensitive Bar)
            vol = self.audio.current_volume
            # Scale up significantly so small sounds are visible
            bar_len = int(vol * 50) 
            cv2.rectangle(frame, (50, 400), (50 + bar_len, 420), (255, 255, 0), -1)
            cv2.putText(frame, f"Vol: {vol:.3f}", (50, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow("Sensitive Acoustic Mouse", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break
                
        cap.release()
        cv2.destroyAllWindows()
        self.audio.stop()

    def mouse_worker(self):
        # Initial wait for vision data
        time.sleep(1.0)
        
        prev_hx = self.target_hand_x
        prev_hy = self.target_hand_y
        last_tap_time = 0
        
        while self.running:
            curr_hx = self.target_hand_x
            curr_hy = self.target_hand_y
            
            # 1. MOVEMENT (Gated by Audio)
            if self.audio.is_sliding and not self.audio.is_tapping:
                # Calculate Delta
                dx = curr_hx - prev_hx
                dy = curr_hy - prev_hy
                
                if abs(dx) > Config.MOVEMENT_DEADZONE or abs(dy) > Config.MOVEMENT_DEADZONE:
                    # Apply Speed & Direction
                    move_x = dx * screen_w * Config.SPEED_MULTIPLIER * Config.DIR_X
                    move_y = dy * screen_h * Config.SPEED_MULTIPLIER * Config.DIR_Y
                    
                    pyautogui.moveRel(move_x, move_y, _pause=False)
                
                # Update Prev to Current
                prev_hx = curr_hx
                prev_hy = curr_hy
                
            else:
                # CLUTCH MODE (Sync Position)
                # When silent, we keep updating prev to curr,
                # so the delta is 0 until we start making noise again.
                prev_hx = curr_hx
                prev_hy = curr_hy
                
                # Debug print for Tap Tuning
                if self.audio.current_volume > 1.0:
                    print(f"NOISE PEAK: {self.audio.current_volume:.2f}")

            # 2. CLICK
            if self.audio.is_tapping:
                if time.time() - last_tap_time > Config.TAP_COOLDOWN:
                    print("--- CLICK FIRED ---")
                    pyautogui.click()
                    last_tap_time = time.time()
            
            time.sleep(0.01)

def main():
    print("--- HIGH SENSITIVITY MOUSE ---")
    print("1. Slide lightly (Hiss) -> Move")
    print("2. Tap lightly (Thud) -> Click")
    
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