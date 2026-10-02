# Gesture Controller

A real-time computer vision application for controlling system functions through hand gestures captured using a webcam.

The application uses MediaPipe for hand landmark detection and gesture recognition, OpenCV for real-time video processing, and PyAutoGUI for system-level mouse and media controls.

## Features

- Real-time hand gesture recognition
- Gesture-based mouse pointer control
- System media playback controls
- System volume control
- Background blur toggle
- Gesture stabilization using temporal smoothing
- Action cooldown to prevent unintended repeated actions
- Live gesture confidence and FPS monitoring
- Desktop GUI for controlling and monitoring the application

## Gesture Mapping

| Gesture | Function |
|---|---|
| Index Finger | Mouse pointer control |
| Pinch | Toggle background blur |
| Open Palm | Play / Pause |
| Thumb Up | Increase volume |
| Thumb Down | Decrease volume |
| Victory | Next track |
| Closed Fist | Previous track |

## Technology Stack

- **Python**
- **OpenCV** — real-time video capture and image processing
- **MediaPipe** — hand landmark detection and gesture recognition
- **PyAutoGUI** — mouse and system media interaction
- **Tkinter** — desktop user interface
- **Pillow** — image handling for the GUI
- **NumPy** — image and mask processing

## Project Structure

```text
gesture_controller/
│
├── app.py
├── camera.py
├── camera_landmarks.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── models/
    ├── hand_landmarker.task
    └── gesture_recognizer.task

How It Works

The application captures live frames from the webcam using OpenCV.

MediaPipe processes each frame to detect hand landmarks and recognize supported gestures. Landmark coordinates are also used for custom interactions such as index-finger mouse control and pinch detection.

To improve reliability, detected gestures are processed over a short frame history before an action is triggered. A cooldown mechanism further reduces unintended repeated actions.

For mouse control, the position of the index fingertip is mapped from the camera coordinate system to the screen coordinate system with smoothing applied to the cursor movement.

Installation
Prerequisites
Python 3.10+
A working webcam
Windows

Clone the repository
git clone https://github.com/sukritidev07/gesture_controller.git
cd gesture_controller
Create a virtual environment
python -m venv venv

Activate the environment

Windows PowerShell:
.\venv\Scripts\Activate.ps1

Windows Command Prompt:
venv\Scripts\activate

Install dependencies
pip install -r requirements.txt
Usage

Run the application with:
python app.py

Position your hand within the camera frame and use the supported gestures to interact with the system.
Mouse control and other gesture-based controls can be enabled or disabled through the application interface.

Reliability
The application uses gesture history and action cooldowns to reduce accidental triggers caused by temporary recognition fluctuations.

Performance may vary depending on:
Lighting conditions
Camera quality
Hand visibility
Distance from the camera
System performance
Future Improvements
Improved person/background segmentation
Mouse calibration
Custom gesture-to-action mapping
Additional accessibility controls
Improved tracking stability
Gesture usage statistics
Expanded gesture vocabulary

Author
Sukriti Roy
B.Tech — Artificial Intelligence & Data Science
GitHub: @sukritidev07
