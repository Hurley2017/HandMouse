import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import math
from collections import deque
import time
import os
import urllib.request

# Disable fail-safe for PyAutoGUI (move mouse to corner to stop)
pyautogui.FAILSAFE = True

# Get screen dimensions
screen_width, screen_height = pyautogui.size()

# Store finger positions for gesture detection
finger_history = deque(maxlen=10)
tap_threshold = 0.05  # Distance threshold for tap detection
min_tap_duration = 0.1  # Minimum time for tap gesture

class HandGestureDetector:
    def __init__(self):
        self.prev_index_tip = None
        self.prev_middle_tip = None
        self.last_tap_time = 0
        self.tap_cooldown = 0.5  # Cooldown between taps (seconds)
        
        # Initialize MediaPipe Hand Landmarker
        model_path = self._ensure_model_downloaded()
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
    
    def _ensure_model_downloaded(self):
        """Download hand landmarker model if not present"""
        model_dir = os.path.expanduser("~/.mediapipe")
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, "hand_landmarker.task")
        
        if not os.path.exists(model_path):
            print("Downloading hand landmarker model... (this may take a moment)")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
            try:
                urllib.request.urlretrieve(url, model_path)
                print(f"Model downloaded to {model_path}")
            except Exception as e:
                print(f"Error downloading model: {e}")
                print("Attempting to use alternative method...")
                return model_path
        
        return model_path
        
    def get_finger_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
    def is_tap_gesture(self, finger_tip, finger_pip, prev_position, gesture_name):
        """
        Detect if a finger performed a tap gesture
        A tap is detected when finger goes down and comes back up
        """
        current_time = time.time()
        
        # Check if enough time has passed since last tap
        if current_time - self.last_tap_time < self.tap_cooldown:
            return False
        
        # Calculate distance between tip and PIP joint
        tip_pip_distance = self.get_finger_distance(finger_tip, finger_pip)
        
        # Tap detected when tip is close to PIP (finger bent down)
        if tip_pip_distance < tap_threshold:
            if prev_position is not None:
                distance_moved = math.sqrt((finger_tip.x - prev_position[0])**2 + 
                                          (finger_tip.y - prev_position[1])**2)
                # Ensure it's a tap (not much movement)
                if distance_moved < 0.1:
                    self.last_tap_time = current_time
                    return True
        
        return False
    
    def process_frame(self, frame, hand_landmarks):
        """Process frame and detect gestures"""
        # Get key landmark points (hand_landmarks is a list of NormalizedLandmark objects)
        # Swapped: 12 for index tip, 10 for index PIP, 8 for middle tip, 6 for middle PIP
        index_tip = hand_landmarks[12]  # Index finger tip (swapped)
        index_pip = hand_landmarks[10]  # Index finger PIP (swapped)
        middle_tip = hand_landmarks[8]   # Middle finger tip (swapped)
        middle_pip = hand_landmarks[6]   # Middle finger PIP (swapped)
        
        # Convert to pixel coordinates
        index_tip_pos = (index_tip.x, index_tip.y)
        middle_tip_pos = (middle_tip.x, middle_tip.y)
        
        # Map hand position to screen - direct mapping (not inverted)
        screen_x = index_tip.x * screen_width
        screen_y = (1 - index_tip.y) * screen_height  # Invert Y-axis for correct up/down mapping
        
        # Move mouse to detected position
        pyautogui.moveTo(int(screen_x), int(screen_y), duration=0.01)
        
        # Detect taps
        if self.is_tap_gesture(index_tip, index_pip, self.prev_index_tip, "Index"):
            pyautogui.click()  # Left click
            print("Left click detected (Index finger tap)")
        
        if self.is_tap_gesture(middle_tip, middle_pip, self.prev_middle_tip, "Middle"):
            pyautogui.rightClick()  # Right click
            print("Right click detected (Middle finger tap)")
        
        # Update previous positions
        self.prev_index_tip = index_tip_pos
        self.prev_middle_tip = middle_tip_pos

def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot access camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    detector = HandGestureDetector()
    
    print("Hand Mouse Control Started!")
    print("Instructions:")
    print("  - Move your right hand to control mouse cursor")
    print("  - Index finger tap (down and up) = Left Click")
    print("  - Middle finger tap (down and up) = Right Click")
    print("  - Press 'q' to quit")
    print("  - Move mouse to top-left corner for fail-safe exit")
    
    while True:
        success, frame = cap.read()
        
        if not success:
            print("Error reading from camera")
            break
        
        # Flip frame horizontally for selfie view
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        
        # Convert to grayscale for faster processing and display
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Convert grayscale back to 3 channels for mediapipe (it will work with grayscale data)
        rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect hand landmarks
        detection_result = detector.landmarker.detect(mp_image)
        
        # Draw hand landmarks on grayscale frame
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                # Draw hand landmarks (white dots on grayscale)
                h_img, w_img = gray_frame.shape
                for landmark in hand_landmarks:
                    x = int(landmark.x * w_img)
                    y = int(landmark.y * h_img)
                    cv2.circle(gray_frame, (x, y), 2, 255, -1)  # White dots
                
                # Process frame for gestures and mouse movement
                detector.process_frame(gray_frame, hand_landmarks)
        
        # Display grayscale frame
        display_frame = cv2.resize(gray_frame, (320, 240))
        cv2.imshow('Hand Mouse Control', display_frame)
        
        # Check for quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Exiting Hand Mouse Control...")
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
