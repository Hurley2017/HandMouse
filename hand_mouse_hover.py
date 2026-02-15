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

# --- CONFIGURATION ---
class Config:
    CAM_WIDTH, CAM_HEIGHT = 640, 480
    FPS = 60 
    
    # --- MICROPHONE SELECTION ---
    # Set this to the ID from the lister script (e.g., 1, 2, etc.)
    # Set to None to use Windows Default
    MIC_DEVICE_ID = 2
    
    # --- DIRECTION CORRECTION ---
    DIR_X = -1 
    DIR_Y = -1
    
    # --- SENSITIVITY ---
    SLIDE_THRESHOLD = 0.08   
    TAP_THRESHOLD = 1.5     
    SPEED_MULTIPLIER = 5.0 
    
    # --- PHYSICS ---
    MOVEMENT_DEADZONE = 1.0 
    TAP_COOLDOWN = 0.25

pyautogui.FAILSAFE = False 
pyautogui.PAUSE = 0 
screen_w, screen_h = pyautogui.size()

class AudioEngine:
    def __init__(self):
        self.running = True
        self.is_sliding = False
        self.is_tapping = False
        self.current_volume = 0.0
        self.calibration_samples = []

    def audio_callback(self, indata, frames, time, status):
        vol = np.linalg.norm(indata) * 10
        self.current_volume = vol
        
        if len(self.calibration_samples) < 50:
            self.calibration_samples.append(vol)
            return
        floor = np.mean(self.calibration_samples)
        
        if vol > Config.TAP_THRESHOLD:
            self.is_tapping = True
            self.is_sliding = True
        elif vol > (floor + Config.SLIDE_THRESHOLD):
            self.is_sliding = True
            self.is_tapping = False
        else:
            self.is_sliding = False
            self.is_tapping = False

    def start(self):
        try:
            print(f"--- ATTEMPTING CONNECTION TO DEVICE ID: {Config.MIC_DEVICE_ID} ---")
            
            # FORCE SPECIFIC DEVICE
            self.stream = sd.InputStream(
                device=Config.MIC_DEVICE_ID, # <--- THE FIX
                callback=self.audio_callback,
                channels=1, 
                samplerate=44100, 
                blocksize=1024
            )
            self.stream.start()
            
            # Confirm which device actually opened
            active_device = self.stream.device
            info = sd.query_devices(active_device)
            print(f"✅ CONNECTED TO: {info['name']}")
            
        except Exception as e:
            print(f"❌ MIC ERROR: {e}")
            print("Fallback to default...")
            try:
                self.stream = sd.InputStream(callback=self.audio_callback, channels=1, samplerate=44100, blocksize=1024)
                self.stream.start()
                print("Connected to Default.")
            except:
                pass

    def stop(self):
        self.running = False
        if self.stream: self.stream.stop(); self.stream.close()

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
        self.audio = AudioEngine()
        self.audio.start()
        
        self.curr_x_px = 0
        self.curr_y_px = 0
        self.axis_forward = np.array([0.0, -1.0]) 
        self.axis_side = np.array([1.0, 0.0])    
        self.running = True

    def _ensure_model_downloaded(self):
        model_path = "hand_landmarker.task"
        if not os.path.exists(model_path):
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
        return model_path

    def update_basis_vectors(self, wrist, middle_mcp):
        wx, wy = wrist.x * Config.CAM_WIDTH, wrist.y * Config.CAM_HEIGHT
        mx, my = middle_mcp.x * Config.CAM_WIDTH, middle_mcp.y * Config.CAM_HEIGHT
        
        vec_x = mx - wx
        vec_y = my - wy 
        
        length = math.hypot(vec_x, vec_y)
        if length < 1e-6: return
        
        ux = vec_x / length
        uy = vec_y / length
        
        vx = -uy
        vy = ux
        
        self.axis_forward = np.array([ux, uy])
        self.axis_side = np.array([vx, vy])

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
                lms = result.hand_landmarks[0]
                idx_tip = lms[8]
                wrist = lms[0]
                mid_mcp = lms[9]
                
                self.update_basis_vectors(wrist, mid_mcp)
                
                self.curr_x_px = idx_tip.x * Config.CAM_WIDTH
                self.curr_y_px = idx_tip.y * Config.CAM_HEIGHT
                
                color = (0, 255, 0) if self.audio.is_sliding else (0, 0, 255)
                cx, cy = int(self.curr_x_px), int(self.curr_y_px)
                cv2.circle(frame, (cx, cy), 8, color, -1)
                
                # Live Axis
                fw_end_x = int(cx + self.axis_forward[0] * 50)
                fw_end_y = int(cy + self.axis_forward[1] * 50)
                cv2.arrowedLine(frame, (cx, cy), (fw_end_x, fw_end_y), (255, 0, 0), 2)
                
                sd_end_x = int(cx + self.axis_side[0] * 30)
                sd_end_y = int(cy + self.axis_side[1] * 30)
                cv2.line(frame, (cx, cy), (sd_end_x, sd_end_y), (0, 0, 255), 2)

            vol = self.audio.current_volume
            bar_w = int(vol * 50)
            cv2.rectangle(frame, (50, 400), (50+bar_w, 420), (255, 255, 0), -1)
            cv2.putText(frame, f"Vol: {vol:.2f}", (50, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow("Acoustic Mouse (Device Selection)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): self.running = False; break
                
        cap.release()
        cv2.destroyAllWindows()
        self.audio.stop()

    def mouse_worker(self):
        time.sleep(1.0) 
        prev_x = self.curr_x_px
        prev_y = self.curr_y_px
        last_tap_time = 0
        
        while self.running:
            curr_x = self.curr_x_px
            curr_y = self.curr_y_px
            
            if self.audio.is_sliding and not self.audio.is_tapping:
                raw_dx = curr_x - prev_x
                raw_dy = curr_y - prev_y
                dist = math.hypot(raw_dx, raw_dy)
                
                if dist > Config.MOVEMENT_DEADZONE:
                    move_forward = -(raw_dx * self.axis_forward[0] + raw_dy * self.axis_forward[1])
                    move_side = (raw_dx * self.axis_side[0] + raw_dy * self.axis_side[1])
                    
                    final_x = move_side * Config.SPEED_MULTIPLIER * Config.DIR_X
                    final_y = -move_forward * Config.SPEED_MULTIPLIER * Config.DIR_Y
                    
                    pyautogui.moveRel(final_x, final_y, _pause=False)
                
                prev_x = curr_x
                prev_y = curr_y
            else:
                prev_x = curr_x
                prev_y = curr_y

            if self.audio.is_tapping:
                if time.time() - last_tap_time > Config.TAP_COOLDOWN:
                    pyautogui.click()
                    last_tap_time = time.time()
            
            time.sleep(0.01)

def main():
    print("--- ACOUSTIC MOUSE (DEVICE SELECT) ---")
    controller = HandController()
    t1 = threading.Thread(target=controller.vision_worker)
    t2 = threading.Thread(target=controller.mouse_worker)
    t1.start(); t2.start(); t1.join(); controller.running = False; t2.join()

if __name__ == "__main__":
    main()