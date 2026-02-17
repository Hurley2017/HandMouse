# HandMouse Project: Evolution & Technical Documentation

## Overview
**HandMouse** is an experimental interface project designed to replace the physical computer mouse with computer vision, acoustic analysis, and sensor fusion. This document outlines the evolutionary steps of the software engine—from basic motion tracking to a physics-based "Neural Trackpad"—and details the future roadmap for a high-precision hardware-based Magnetic Interface.

---

## Phase I: The Software Evolution (Computer Vision & Acoustics)

The following versions represent the iterative development of the `HandMouse` engine, solving specific problems of jitter, ergonomics, and reliability.

### 1. The Acoustic Prototype (Absolute Positioning)
**Concept:** A hybrid input using the microphone as a "Clutch" and the camera for cursor mapping.
* **Mechanism:**
    * **Move:** User makes a continuous "Hissing" sound (imitating sliding friction) to engage tracking. Silence freezes the cursor.
    * **Click:** A sharp "Thud" on the desk triggers a click via audio amplitude spike detection.
* **Failure Point:** Used **Absolute Mapping** (Camera Pixel $X$ = Screen Pixel $X$).
* **Outcome:** High jitter. Users had to hold their hand perfectly still to prevent cursor drift.

### 2. The "Delta" Engine (Relative Movement)
**Concept:** Shifted from "Map to Screen" to "Map to Motion."
* **Innovation:** **Relative Delta Tracking**.
    * Instead of tracking *position*, the engine calculates *velocity* ($dX, dY$).
* **The "Reset Glitch" Fix:**
    * *Problem:* In V1, lifting the hand (silence) and moving it back caused the cursor to snap back to the center.
    * *Solution:* The **Shadow Tracker**. The "previous position" variable is continuously updated even during silence (Clutch Mode). This allows the user to "ratchet" their hand like a physical mouse without the cursor jumping.

### 3. The Ergonomic Pivot (Coordinate Rotation)
**Concept:** Addressed the biomechanical reality that users rarely hold their arms at a perfect 90° angle to the camera.
* **Problem:** If the arm is angled at 45°, moving the hand "forward" (relative to the arm) resulted in a diagonal cursor movement.
* **Solution:** **Calibration on Start**.
    * The system calculates the angle $\theta$ between the Wrist and Middle Finger.
    * A **2D Rotation Matrix** is applied to all subsequent input vectors to align "Hand Forward" with "Screen Up."

### 4. The Live Axis Engine (Continuous Re-Orientation)
**Concept:** Removed the need for static calibration by calculating orientation every single frame.
* **Mechanism:**
    * Constructs a **Live Vector** from Wrist to Middle Finger Knuckle ($V_{fwd}$).
    * Constructs a perpendicular **Side Vector** ($V_{side}$).
* **The Math:** Uses **Dot Product Projection**.
    * Movement is calculated as: *"How much of the vector was parallel to the finger?"* and *"How much was perpendicular?"*
    * 

[Image of vector projection diagram]

* **Outcome:** User can rotate their chair, arm, or keyboard freely during use without recalibrating.

### 5. The Virtual Trackpad (Gesture Engine)
**Concept:** Abandoned audio input for a purely geometric "Computer Vision" approach to improve reliability in noisy environments.
* **State Machine:**
    * **Move:** Index Finger Extended.
    * **Scroll:** Index + Middle Finger Extended (Peace Sign).
    * **Clutch:** Fist (Curled Fingers).
* **Visual Feedback:** The cursor on the camera feed changes color (Green=Move, Magenta=Scroll, Red=Clutch) to confirm state.

### 6. The "Neural" Final Polish (Physics & Physiology)
**Concept:** The definitive software version, focusing on "feel" and physiological stability.
* **Physiological Clicks:** Replaced "Pinch" gestures (which pull the index finger and cause jitter) with muscle-isolated gestures.
    * **Left Click:** **Thumb Tuck** (Tucking thumb to index knuckle using the adductor muscle).
    * **Right Click:** **Middle Trigger** (Curling middle finger independently).
* **Physics Engine:**
    * **Deadzone:** Ignores micro-tremors under 1.2 pixels.
    * **Smoothing:** Averages motion over 4 frames for weighted inertia.
    * **Inertia:** Scrolling carries momentum for a natural, smartphone-like feel.

---

## Phase II: Future Roadmap (The Magnetic Interface)

While the camera-based "Neural Trackpad" is software-complete, it suffers from inherent limitations of optical tracking (lighting conditions, camera frame rate, and CPU latency). The next evolution proposes a **Hardware-Based Approach**.

### The Concept: 3D Magnetic Flux Tracking
Instead of "seeing" the hand with a camera, the system senses the invisible magnetic field of a passive ring worn by the user.

### Why It's Superior
1.  **Infinite Precision:** Magnetic fields change smoothly and continuously, offering sub-millimeter precision that exceeds 4K pixel grids.
2.  **Zero Light Dependency:** Works in pitch darkness or under desk surfaces.
3.  **True Z-Axis (Hover):** A 3D Hall Effect sensor can measure exactly how high the ring is hovering (e.g., 5mm vs 50mm), allowing for "Pressure-Sensitive Hovering."

### Hardware Implementation Stack
* **The Ring:** A simple **Neodymium Magnet** embedded in a ring worn on the middle finger.
* **The Sensor:** **MLX90393 (3D Hall Effect)**.
    * Capable of reading magnetic flux ($B_x, B_y, B_z$) in micro-Teslas.
    * 

[Image of hall effect sensor principle]

* **Controller:** **ESP32** or **Arduino Pro Micro** (acting as a native USB HID Mouse).

### The "Dual-Sensor" Array Strategy
To create a large "Active Area" (like a Wacom tablet) without dead zones, the proposed design uses **Two Sensors** spaced 10cm apart.
* **Triangulation:** The relative strength of the magnetic field between Sensor A and Sensor B calculates the exact 2D position.
* **Clicking:** A sharp spike in the Z-axis flux (tapping the desk) is detected as a click, eliminating the need for physical buttons.