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
