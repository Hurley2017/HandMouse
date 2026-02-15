# HandMouse - Hand Gesture Mouse Control

A Python utility that uses your webcam to control your mouse with hand gestures and finger taps.

## Features

- **Mouse Control**: Move your right hand to control the mouse cursor in real-time
- **Left Click**: Tap your index finger (down and up motion) for left click
- **Right Click**: Tap your middle finger (down and up motion) for right click
- **Real-time Hand Detection**: Uses MediaPipe for accurate hand pose estimation
- **Smooth Movement**: Natural mouse movement tracking

## Requirements

- Python 3.8 or higher
- Webcam
- Windows/Mac/Linux

## Installation

### 1. Navigate to project directory
```bash
cd HandMouse
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
```

### 3. Activate virtual environment
**On Windows:**
```bash
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
source venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
python hand_mouse.py
```

### Controls

- **Mouse Movement**: Move your right hand in front of the camera. Your index finger tip position controls the cursor.
- **Left Click**: Quickly tap your index finger (bend and straighten it)
- **Right Click**: Quickly tap your middle finger (bend and straighten it)
- **Exit**: Press 'q' or move your mouse to the top-left corner of the screen (failsafe)

## How It Works

1. **Hand Detection**: Uses MediaPipe Hands to detect hand landmarks in real-time
2. **Position Mapping**: Maps your hand position to screen coordinates
3. **Gesture Recognition**: Detects finger taps by monitoring the distance between finger tip and PIP joint
4. **Mouse Control**: Uses PyAutoGUI to control mouse movement and clicks

## Gesture Details

- **Tap Gesture**: A quick downward motion of the finger followed by release. The finger should move down (decreasing tip-to-PIP distance) and then return to normal position.
- **Cooldown**: There's a 0.5 second cooldown between clicks to prevent accidental double-clicks

## Troubleshooting

### Camera not detected
- Ensure your webcam is connected and not in use by another application
- Check if the camera has proper permissions

### Gestures not registering
- Make sure you perform a clear downward tap motion
- Ensure good lighting for better hand detection
- Increase the `tap_threshold` value in the code if taps are too sensitive

### Cursor movement is shaky
- Ensure stable lighting conditions
- Move more slowly
- Reduce `duration` parameter in `pyautogui.moveTo()` for faster response

## Configuration

Edit `hand_mouse.py` to adjust:
- `tap_threshold`: Distance threshold for tap detection (default: 0.05)
- `min_detection_confidence`: Hand detection confidence (0-1, default: 0.7)
- `tap_cooldown`: Time between clicks in seconds (default: 0.5)

## Safety Features

- **Failsafe**: Move mouse to top-left corner of screen to stop the program
- **Press 'q'**: Exit the application anytime

## Limitations

- Currently works with right hand only
- Requires good lighting for optimal detection
- Works best on a white or plain background
- Finger taps require quick and distinct motion

## Future Improvements

- Support for left hand
- Both hands support with dual cursor mode
- Additional gestures (palm open, thumb tap)
- Configuration file for easy customization
- Gesture recording and custom gesture creation

## License

MIT License
