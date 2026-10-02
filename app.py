import cv2
import mediapipe as mp
import pyautogui
import tkinter as tk
from PIL import Image, ImageTk
import threading
import time
import math
import numpy as np
from collections import deque


# ============================================================
# COLORS
# ============================================================

BG = "#090d12"
PANEL = "#111820"
GREEN = "#00e6a0"
YELLOW = "#ffbd45"
RED = "#ff5c70"
WHITE = "#f4f7fa"
GREY = "#8b98a5"


# ============================================================
# SETTINGS
# ============================================================

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

HISTORY_SIZE = 7
STABLE_REQUIRED = 5

ACTION_COOLDOWN = 1.2

# Pinch sensitivity
PINCH_THRESHOLD = 0.45

# Mouse
MOUSE_SMOOTHING = 0.25
MOUSE_MARGIN = 0.10

# Blur
BLUR_AMOUNT = 31
HAND_MASK_SIZE = 55


# ============================================================
# MOUSE SETTINGS
# ============================================================

pyautogui.PAUSE = 0.01

screen_width, screen_height = pyautogui.size()


# ============================================================
# GLOBAL STATE
# ============================================================

running = True

controller_enabled = True
mouse_control_enabled = True
blur_enabled = False

current_gesture = "None"
current_confidence = 0.0
fps_value = 0

gesture_history = deque(
    maxlen=HISTORY_SIZE
)

last_action_time = 0

pinch_active = False

mouse_x = screen_width // 2
mouse_y = screen_height // 2

latest_frame = None

state_lock = threading.Lock()


# ============================================================
# GESTURE NAMES
# ============================================================

gesture_names = {
    "Open_Palm": "Open Palm",
    "Thumb_Up": "Thumb Up",
    "Thumb_Down": "Thumb Down",
    "Victory": "Victory",
    "Closed_Fist": "Closed Fist",
    "Index_Finger": "Index Finger",
    "Pinch": "Pinch",
    "None": "None"
}


# ============================================================
# PINCH DETECTION
# ============================================================

def is_pinch(hand):

    thumb_tip = hand[4]
    index_tip = hand[8]

    wrist = hand[0]
    middle_base = hand[9]

    distance = math.sqrt(
        (thumb_tip.x - index_tip.x) ** 2 +
        (thumb_tip.y - index_tip.y) ** 2
    )

    hand_size = math.sqrt(
        (wrist.x - middle_base.x) ** 2 +
        (wrist.y - middle_base.y) ** 2
    )

    if hand_size == 0:
        return False

    normalized_distance = (
        distance / hand_size
    )

    index_extended = (
        hand[8].y < hand[6].y
    )

    return (
        normalized_distance < PINCH_THRESHOLD
        and index_extended
    )


# ============================================================
# INDEX FINGER DETECTION
# ============================================================

def is_index_pointing(hand):

    index_extended = (
        hand[8].y < hand[6].y
    )

    middle_folded = (
        hand[12].y > hand[10].y
    )

    ring_folded = (
        hand[16].y > hand[14].y
    )

    pinky_folded = (
        hand[20].y > hand[18].y
    )

    return (
        index_extended
        and middle_folded
        and ring_folded
        and pinky_folded
    )


# ============================================================
# MOUSE MOVEMENT
# ============================================================

def move_mouse(index_finger):

    global mouse_x
    global mouse_y

    x = index_finger.x
    y = index_finger.y

    # Camera movement area
    x = (
        x - MOUSE_MARGIN
    ) / (
        1 - 2 * MOUSE_MARGIN
    )

    y = (
        y - MOUSE_MARGIN
    ) / (
        1 - 2 * MOUSE_MARGIN
    )

    # Keep inside screen
    x = max(
        0,
        min(1, x)
    )

    y = max(
        0,
        min(1, y)
    )

    target_x = (
        x * screen_width
    )

    target_y = (
        y * screen_height
    )

    # Smooth movement
    mouse_x += (
        target_x - mouse_x
    ) * MOUSE_SMOOTHING

    mouse_y += (
        target_y - mouse_y
    ) * MOUSE_SMOOTHING

    pyautogui.moveTo(
        int(mouse_x),
        int(mouse_y),
        duration=0
    )


# ============================================================
# MEDIA GESTURES
# ============================================================

def perform_gesture(gesture):

    global last_action_time

    now = time.time()

    if (
        now - last_action_time
        < ACTION_COOLDOWN
    ):
        return

    if gesture == "Open_Palm":

        pyautogui.press(
            "playpause"
        )

    elif gesture == "Thumb_Up":

        pyautogui.press(
            "volumeup"
        )

    elif gesture == "Thumb_Down":

        pyautogui.press(
            "volumedown"
        )

    elif gesture == "Victory":

        pyautogui.press(
            "nexttrack"
        )

    elif gesture == "Closed_Fist":

        pyautogui.press(
            "prevtrack"
        )

    else:

        return

    last_action_time = now


# ============================================================
# BACKGROUND BLUR
# ============================================================

def blur_background(frame, hand):

    blurred = cv2.GaussianBlur(
        frame,
        (
            BLUR_AMOUNT,
            BLUR_AMOUNT
        ),
        0
    )

    h, w, _ = frame.shape

    points = []

    for landmark in hand:

        x = int(
            landmark.x * w
        )

        y = int(
            landmark.y * h
        )

        points.append(
            [x, y]
        )

    if len(points) < 3:

        return blurred

    points = np.array(
        points,
        dtype=np.int32
    )

    hull = cv2.convexHull(
        points
    )

    mask = np.zeros(
        (h, w),
        dtype=np.uint8
    )

    cv2.fillConvexPoly(
        mask,
        hull,
        255
    )

    # Expand around the hand
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            HAND_MASK_SIZE,
            HAND_MASK_SIZE
        )
    )

    mask = cv2.dilate(
        mask,
        kernel
    )

    # Smooth edges
    mask = cv2.GaussianBlur(
        mask,
        (31, 31),
        0
    )

    mask = (
        mask.astype(np.float32)
        / 255.0
    )

    mask = mask[:, :, None]

    original = frame.astype(
        np.float32
    )

    blurred_float = blurred.astype(
        np.float32
    )

    result = (
        original * mask
        +
        blurred_float *
        (1 - mask)
    )

    return result.astype(
        np.uint8
    )


# ============================================================
# CAMERA THREAD
# ============================================================

def camera_loop():

    global running
    global current_gesture
    global current_confidence
    global fps_value
    global pinch_active
    global latest_frame
    global blur_enabled

    # ========================================================
    # CREATE MEDIAPIPE INSIDE CAMERA THREAD
    # ========================================================

    BaseOptions = mp.tasks.BaseOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    GestureRecognizer = (
        mp.tasks.vision.GestureRecognizer
    )

    GestureRecognizerOptions = (
        mp.tasks.vision.GestureRecognizerOptions
    )

    options = GestureRecognizerOptions(
        base_options=BaseOptions(
            model_asset_path=(
                "models/gesture_recognizer.task"
            )
        ),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1
    )

    recognizer = (
        GestureRecognizer.create_from_options(
            options
        )
    )

    # ========================================================
    # CAMERA
    # ========================================================

    cap = cv2.VideoCapture(0)

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT
    )

    previous_time = time.time()

    try:

        while running:

            success, frame = cap.read()

            if not success:

                time.sleep(0.05)

                continue

            # =================================================
            # MIRROR
            # =================================================

            frame = cv2.flip(
                frame,
                1
            )

            # =================================================
            # FPS
            # =================================================

            now = time.time()

            elapsed = (
                now - previous_time
            )

            if elapsed > 0:

                with state_lock:

                    fps_value = int(
                        1 / elapsed
                    )

            previous_time = now

            # =================================================
            # MEDIAPIPE
            # =================================================

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=(
                    mp.ImageFormat.SRGB
                ),
                data=rgb
            )

            result = recognizer.recognize(
                mp_image
            )

            hand_detected = bool(
                result.hand_landmarks
            )

            # =================================================
            # HAND DETECTED
            # =================================================

            if hand_detected:

                hand = result.hand_landmarks[0]

                with state_lock:

                    enabled = (
                        controller_enabled
                    )

                    mouse_enabled = (
                        mouse_control_enabled
                    )

                if enabled:

                    # =========================================
                    # PINCH
                    # =========================================

                    pinch_now = is_pinch(
                        hand
                    )

                    if pinch_now:

                        with state_lock:

                            current_gesture = (
                                "Pinch"
                            )

                            current_confidence = (
                                1.0
                            )

                        # Toggle blur only once
                        if not pinch_active:

                            with state_lock:

                                blur_enabled = (
                                    not blur_enabled
                                )

                        pinch_active = True

                    else:

                        pinch_active = False

                        # =====================================
                        # INDEX FINGER = MOUSE
                        # =====================================

                        if is_index_pointing(
                            hand
                        ):

                            with state_lock:

                                current_gesture = (
                                    "Index_Finger"
                                )

                                current_confidence = (
                                    1.0
                                )

                            if mouse_enabled:

                                move_mouse(
                                    hand[8]
                                )

                            gesture_history.clear()

                        else:

                            # =================================
                            # BUILT-IN GESTURES
                            # =================================

                            if result.gestures:

                                detected = (
                                    result.gestures[0][0]
                                )

                                name = (
                                    detected.category_name
                                )

                                confidence = (
                                    detected.score
                                )

                                if (
                                    name
                                    not in gesture_names
                                ):

                                    name = "None"

                                # Update UI immediately
                                with state_lock:

                                    current_gesture = (
                                        name
                                    )

                                    current_confidence = (
                                        confidence
                                    )

                                # Add to history
                                gesture_history.append(
                                    name
                                )

                                # Stable action
                                if (
                                    len(
                                        gesture_history
                                    )
                                    >= STABLE_REQUIRED
                                ):

                                    recent = list(
                                        gesture_history
                                    )

                                    most_common = max(
                                        set(recent),
                                        key=recent.count
                                    )

                                    count = (
                                        recent.count(
                                            most_common
                                        )
                                    )

                                    if (
                                        count
                                        >= STABLE_REQUIRED
                                    ):

                                        perform_gesture(
                                            most_common
                                        )

                            else:

                                with state_lock:

                                    current_gesture = (
                                        "None"
                                    )

                                    current_confidence = (
                                        0.0
                                    )

                else:

                    pinch_active = False

                    with state_lock:

                        current_gesture = (
                            "Disabled"
                        )

                        current_confidence = (
                            0.0
                        )

            else:

                gesture_history.clear()

                pinch_active = False

                with state_lock:

                    current_gesture = (
                        "None"
                    )

                    current_confidence = (
                        0.0
                    )

            # =================================================
            # BACKGROUND BLUR
            # =================================================

            with state_lock:

                do_blur = (
                    blur_enabled
                )

            if (
                do_blur
                and hand_detected
            ):

                frame = blur_background(
                    frame,
                    result.hand_landmarks[0]
                )

            # =================================================
            # DRAW LANDMARKS
            # =================================================

            if hand_detected:

                hand = (
                    result.hand_landmarks[0]
                )

                h, w, _ = frame.shape

                for landmark in hand:

                    px = int(
                        landmark.x * w
                    )

                    py = int(
                        landmark.y * h
                    )

                    cv2.circle(
                        frame,
                        (px, py),
                        5,
                        (0, 230, 160),
                        -1
                    )

                # Index fingertip
                ix = int(
                    hand[8].x * w
                )

                iy = int(
                    hand[8].y * h
                )

                cv2.circle(
                    frame,
                    (ix, iy),
                    12,
                    (255, 189, 69),
                    2
                )

            # =================================================
            # CAMERA OVERLAY
            # =================================================

            cv2.putText(
                frame,
                "GESTURE CONTROLLER",
                (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 230, 160),
                2
            )

            with state_lock:

                enabled = (
                    controller_enabled
                )

                do_blur = (
                    blur_enabled
                )

            if enabled:

                cv2.putText(
                    frame,
                    "CONTROLS ENABLED",
                    (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 230, 160),
                    2
                )

            else:

                cv2.putText(
                    frame,
                    "CONTROLS DISABLED",
                    (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 92, 112),
                    2
                )

            if do_blur:

                cv2.putText(
                    frame,
                    "BACKGROUND BLUR: ON",
                    (30, 115),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 189, 69),
                    2
                )

            # =================================================
            # SAVE FRAME
            # =================================================

            with state_lock:

                latest_frame = (
                    frame.copy()
                )

            time.sleep(0.005)

    finally:

        cap.release()

        try:

            recognizer.close()

        except:

            pass


# ============================================================
# UPDATE CAMERA PREVIEW
# ============================================================

def update_preview():

    if not running:

        return

    with state_lock:

        if latest_frame is not None:

            frame = latest_frame.copy()

        else:

            frame = None

    if frame is not None:

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(
            rgb
        )

        image.thumbnail(
            (900, 600),
            Image.Resampling.LANCZOS
        )

        photo = ImageTk.PhotoImage(
            image
        )

        camera_label.configure(
            image=photo
        )

        camera_label.image = photo

    root.after(
        30,
        update_preview
    )


# ============================================================
# UPDATE GUI
# ============================================================

def update_gui():

    if not running:

        return

    # ========================================================
    # READ STATE
    # ========================================================

    with state_lock:

        gesture = current_gesture
        confidence = current_confidence
        fps = fps_value

        enabled = (
            controller_enabled
        )

        mouse_enabled = (
            mouse_control_enabled
        )

        do_blur = (
            blur_enabled
        )

    # ========================================================
    # CURRENT GESTURE
    # ========================================================

    if gesture == "Index_Finger":

        display = (
            "INDEX FINGER → MOUSE"
        )

    elif gesture == "Pinch":

        if do_blur:

            display = (
                "PINCH → BLUR ON"
            )

        else:

            display = (
                "PINCH → BLUR OFF"
            )

    elif gesture == "Disabled":

        display = (
            "CONTROLS OFF"
        )

    else:

        display = gesture_names.get(
            gesture,
            gesture
        )

    gesture_value.config(
        text=display
    )

    # ========================================================
    # CONFIDENCE
    # ========================================================

    percent = int(
        confidence * 100
    )

    confidence_value.config(
        text=f"{percent}%"
    )

    confidence_bar.set(
        percent
    )

    # ========================================================
    # STATUS
    # ========================================================

    if enabled:

        status_value.config(
            text="● ACTIVE",
            fg=GREEN
        )

        enable_button.config(
            text="DISABLE CONTROLS",
            fg=GREEN
        )

    else:

        status_value.config(
            text="● DISABLED",
            fg=RED
        )

        enable_button.config(
            text="ENABLE CONTROLS",
            fg=RED
        )

    # ========================================================
    # MOUSE STATUS
    # ========================================================

    if mouse_enabled:

        mouse_button.config(
            text="MOUSE: ON",
            fg=GREEN
        )

    else:

        mouse_button.config(
            text="MOUSE: OFF",
            fg=RED
        )

    # ========================================================
    # BLUR STATUS
    # ========================================================

    if do_blur:

        blur_button.config(
            text="BLUR: ON",
            fg=YELLOW
        )

    else:

        blur_button.config(
            text="BLUR: OFF",
            fg=GREY
        )

    # ========================================================
    # FPS
    # ========================================================

    fps_value_label.config(
        text=str(fps)
    )

    root.after(
        100,
        update_gui
    )


# ============================================================
# CONTROLLER TOGGLE
# ============================================================

def toggle_controller():

    global controller_enabled

    with state_lock:

        controller_enabled = (
            not controller_enabled
        )

    gesture_history.clear()


# ============================================================
# MOUSE TOGGLE
# ============================================================

def toggle_mouse():

    global mouse_control_enabled

    with state_lock:

        mouse_control_enabled = (
            not mouse_control_enabled
        )

    gesture_history.clear()


# ============================================================
# BLUR BUTTON
# ============================================================

def toggle_blur_direct():

    global blur_enabled

    with state_lock:

        blur_enabled = (
            not blur_enabled
        )


# ============================================================
# QUIT
# ============================================================

def quit_app():

    global running

    running = False

    root.destroy()


# ============================================================
# TKINTER WINDOW
# ============================================================

root = tk.Tk()

root.title(
    "Gesture Controller"
)

root.geometry(
    "1280x850"
)

root.minsize(
    1100,
    720
)

root.configure(
    bg=BG
)

root.protocol(
    "WM_DELETE_WINDOW",
    quit_app
)


# ============================================================
# HEADER
# ============================================================

header = tk.Frame(
    root,
    bg=BG
)

header.pack(
    fill="x",
    padx=30,
    pady=(20, 10)
)


tk.Label(
    header,
    text="GESTURE CONTROLLER",
    font=("Segoe UI", 26, "bold"),
    fg=WHITE,
    bg=BG
).pack(
    side="left"
)


tk.Label(
    header,
    text="AI-POWERED HAND INTERFACE",
    font=("Segoe UI", 10, "bold"),
    fg=GREEN,
    bg=BG
).pack(
    side="left",
    padx=18,
    pady=(8, 0)
)


# ============================================================
# MAIN CONTAINER
# ============================================================

main = tk.Frame(
    root,
    bg=BG
)

main.pack(
    fill="both",
    expand=True,
    padx=30,
    pady=10
)


# ============================================================
# CAMERA PANEL
# ============================================================

camera_panel = tk.Frame(
    main,
    bg=PANEL
)

camera_panel.pack(
    side="left",
    fill="both",
    expand=True,
    padx=(0, 10)
)


camera_header = tk.Frame(
    camera_panel,
    bg=PANEL
)

camera_header.pack(
    fill="x",
    padx=20,
    pady=15
)


tk.Label(
    camera_header,
    text="LIVE CAMERA",
    font=("Segoe UI", 12, "bold"),
    fg=WHITE,
    bg=PANEL
).pack(
    side="left"
)


tk.Label(
    camera_header,
    text="● HAND TRACKING ENABLED",
    font=("Segoe UI", 9),
    fg=GREEN,
    bg=PANEL
).pack(
    side="right"
)


camera_container = tk.Frame(
    camera_panel,
    bg="#05080b"
)

camera_container.pack(
    fill="both",
    expand=True,
    padx=15,
    pady=(0, 15)
)


camera_label = tk.Label(
    camera_container,
    bg="#05080b"
)

camera_label.pack(
    expand=True
)


# ============================================================
# RIGHT PANEL
# ============================================================

right_panel = tk.Frame(
    main,
    bg=BG,
    width=300
)

right_panel.pack(
    side="right",
    fill="y"
)

right_panel.pack_propagate(
    False
)


# ============================================================
# CURRENT GESTURE CARD
# ============================================================

gesture_card = tk.Frame(
    right_panel,
    bg=PANEL
)

gesture_card.pack(
    fill="x",
    pady=(0, 10)
)


tk.Label(
    gesture_card,
    text="CURRENT GESTURE",
    font=("Segoe UI", 9, "bold"),
    fg=GREY,
    bg=PANEL
).pack(
    anchor="w",
    padx=18,
    pady=(15, 5)
)


gesture_value = tk.Label(
    gesture_card,
    text="None",
    font=("Segoe UI", 17, "bold"),
    fg=GREEN,
    bg=PANEL
)

gesture_value.pack(
    anchor="w",
    padx=18,
    pady=(0, 15)
)


# ============================================================
# STATUS CARD
# ============================================================

status_card = tk.Frame(
    right_panel,
    bg=PANEL
)

status_card.pack(
    fill="x",
    pady=(0, 10)
)


tk.Label(
    status_card,
    text="DETECTION STATUS",
    font=("Segoe UI", 9, "bold"),
    fg=GREY,
    bg=PANEL
).pack(
    anchor="w",
    padx=18,
    pady=(15, 5)
)


status_value = tk.Label(
    status_card,
    text="● ACTIVE",
    font=("Segoe UI", 13, "bold"),
    fg=GREEN,
    bg=PANEL
)

status_value.pack(
    anchor="w",
    padx=18
)


tk.Label(
    status_card,
    text="CONFIDENCE",
    font=("Segoe UI", 9, "bold"),
    fg=GREY,
    bg=PANEL
).pack(
    anchor="w",
    padx=18,
    pady=(15, 4)
)


confidence_value = tk.Label(
    status_card,
    text="0%",
    font=("Segoe UI", 12, "bold"),
    fg=WHITE,
    bg=PANEL
)

confidence_value.pack(
    anchor="w",
    padx=18
)


confidence_bar = tk.Scale(
    status_card,
    from_=0,
    to=100,
    orient="horizontal",
    showvalue=False,
    state="normal",
    bg=PANEL,
    fg=GREEN,
    troughcolor="#26323c",
    highlightthickness=0,
    bd=0,
    sliderlength=1
)

confidence_bar.pack(
    fill="x",
    padx=18,
    pady=(3, 15)
)


# ============================================================
# GESTURE MAP
# ============================================================

map_card = tk.Frame(
    right_panel,
    bg=PANEL
)

map_card.pack(
    fill="x",
    pady=(0, 10)
)


tk.Label(
    map_card,
    text="GESTURE MAP",
    font=("Segoe UI", 9, "bold"),
    fg=GREY,
    bg=PANEL
).pack(
    anchor="w",
    padx=18,
    pady=(15, 10)
)


gesture_map = [
    ("☝", "Index Finger", "Mouse"),
    ("🤏", "Pinch", "Blur ON / OFF"),
    ("✋", "Open Palm", "Play / Pause"),
    ("👍", "Thumb Up", "Volume +"),
    ("👎", "Thumb Down", "Volume -"),
    ("✌", "Victory", "Next Track"),
    ("✊", "Fist", "Previous Track"),
]


for icon, gesture, action in gesture_map:

    row = tk.Frame(
        map_card,
        bg=PANEL
    )

    row.pack(
        fill="x",
        padx=15,
        pady=3
    )

    tk.Label(
        row,
        text=icon,
        font=("Segoe UI Emoji", 13),
        fg=WHITE,
        bg=PANEL,
        width=3
    ).pack(
        side="left"
    )

    tk.Label(
        row,
        text=gesture,
        font=("Segoe UI", 9, "bold"),
        fg=WHITE,
        bg=PANEL
    ).pack(
        side="left"
    )

    tk.Label(
        row,
        text=action,
        font=("Segoe UI", 8),
        fg=GREY,
        bg=PANEL
    ).pack(
        side="right"
    )


# ============================================================
# PREVIEW CONTROLS
# ============================================================

controls_card = tk.Frame(
    right_panel,
    bg=PANEL
)

controls_card.pack(
    fill="x",
    pady=(0, 10)
)


tk.Label(
    controls_card,
    text="PREVIEW",
    font=("Segoe UI", 9, "bold"),
    fg=GREY,
    bg=PANEL
).pack(
    anchor="w",
    padx=18,
    pady=(12, 8)
)


# ============================================================
# BLUR BUTTON
# ============================================================

blur_button = tk.Button(
    controls_card,
    text="BLUR: OFF",
    command=toggle_blur_direct,
    font=("Segoe UI", 8, "bold"),
    bg="#1c2b35",
    fg=GREY,
    activebackground="#263944",
    activeforeground=YELLOW,
    relief="flat",
    bd=0,
    padx=12,
    pady=8,
    cursor="hand2"
)

blur_button.pack(
    fill="x",
    padx=15,
    pady=3
)


# ============================================================
# MOUSE BUTTON
# ============================================================

mouse_button = tk.Button(
    controls_card,
    text="MOUSE: ON",
    command=toggle_mouse,
    font=("Segoe UI", 8, "bold"),
    bg="#1c2b35",
    fg=GREEN,
    activebackground="#263944",
    activeforeground=GREEN,
    relief="flat",
    bd=0,
    padx=12,
    pady=8,
    cursor="hand2"
)

mouse_button.pack(
    fill="x",
    padx=15,
    pady=3
)


# ============================================================
# BOTTOM BAR
# ============================================================

bottom = tk.Frame(
    root,
    bg=BG
)

bottom.pack(
    fill="x",
    padx=30,
    pady=(5, 20)
)


# ============================================================
# FPS
# ============================================================

fps_box = tk.Frame(
    bottom,
    bg=PANEL
)

fps_box.pack(
    side="left",
    padx=(0, 10)
)


tk.Label(
    fps_box,
    text="FPS",
    font=("Segoe UI", 8, "bold"),
    fg=GREY,
    bg=PANEL
).pack(
    side="left",
    padx=(12, 4),
    pady=10
)


fps_value_label = tk.Label(
    fps_box,
    text="0",
    font=("Segoe UI", 10, "bold"),
    fg=GREEN,
    bg=PANEL
)

fps_value_label.pack(
    side="left",
    padx=(0, 12)
)


# ============================================================
# ENABLE CONTROLS
# ============================================================

enable_button = tk.Button(
    bottom,
    text="DISABLE CONTROLS",
    command=toggle_controller,
    font=("Segoe UI", 9, "bold"),
    bg="#1c2b35",
    fg=GREEN,
    activebackground="#263944",
    activeforeground=GREEN,
    relief="flat",
    bd=0,
    padx=20,
    pady=10,
    cursor="hand2"
)

enable_button.pack(
    side="right",
    padx=5
)


# ============================================================
# QUIT
# ============================================================

quit_button = tk.Button(
    bottom,
    text="QUIT",
    command=quit_app,
    font=("Segoe UI", 9, "bold"),
    bg="#301820",
    fg=RED,
    activebackground="#40212a",
    activeforeground=RED,
    relief="flat",
    bd=0,
    padx=25,
    pady=10,
    cursor="hand2"
)

quit_button.pack(
    side="right",
    padx=5
)


# ============================================================
# START CAMERA THREAD
# ============================================================

camera_thread = threading.Thread(
    target=camera_loop,
    daemon=True
)

camera_thread.start()


# ============================================================
# START GUI LOOPS
# ============================================================

root.after(
    30,
    update_preview
)

root.after(
    100,
    update_gui
)


# ============================================================
# START APPLICATION
# ============================================================

root.mainloop()


# ============================================================
# FINAL CLEANUP
# ============================================================

running = False