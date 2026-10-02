import cv2
import mediapipe as mp
import pyautogui
import time
import math
from collections import deque, Counter


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "models/gesture_recognizer.task"

MIN_CONFIDENCE = 0.65

HISTORY_SIZE = 7

ACTION_COOLDOWN = 1.2

# Pinch sensitivity
PINCH_RATIO = 0.38

# Swipe settings
SWIPE_DISTANCE = 0.18
SWIPE_TIME = 0.7
SWIPE_COOLDOWN = 1.0


# ============================================================
# MEDIAPIPE
# ============================================================

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


options = GestureRecognizerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=MIN_CONFIDENCE,
    min_hand_presence_confidence=MIN_CONFIDENCE,
    min_tracking_confidence=MIN_CONFIDENCE
)


# ============================================================
# HISTORY
# ============================================================

gesture_history = deque(maxlen=HISTORY_SIZE)

wrist_history = deque(maxlen=30)

last_action = None
last_action_time = 0

last_swipe_time = 0


# ============================================================
# DISTANCE FUNCTION
# ============================================================

def distance(point1, point2):

    return math.sqrt(
        (point1.x - point2.x) ** 2 +
        (point1.y - point2.y) ** 2
    )


# ============================================================
# PINCH DETECTION
# ============================================================

def is_pinch(hand):

    thumb_tip = hand[4]
    index_tip = hand[8]

    wrist = hand[0]
    middle_base = hand[9]

    # Distance between thumb and index
    thumb_index_distance = distance(
        thumb_tip,
        index_tip
    )

    # Size of the hand
    hand_size = distance(
        wrist,
        middle_base
    )

    if hand_size == 0:
        return False

    ratio = (
        thumb_index_distance /
        hand_size
    )

    # Index finger must actually be extended.
    # This prevents a closed fist from being
    # incorrectly detected as a pinch.
    index_extended = (
        hand[8].y <
        hand[6].y
    )

    if ratio < PINCH_RATIO and index_extended:

        return True

    return False


# ============================================================
# STABLE GESTURE
# ============================================================

def get_stable_gesture(current_gesture):

    gesture_history.append(current_gesture)

    if len(gesture_history) < HISTORY_SIZE:

        return "Stabilizing..."

    counts = Counter(gesture_history)

    gesture, count = counts.most_common(1)[0]

    if count >= 5:

        return gesture

    return "UNKNOWN"


# ============================================================
# NORMAL GESTURE ACTIONS
# ============================================================

GESTURE_ACTIONS = {

    "Open_Palm":
        ("PLAY / PAUSE", "playpause"),

    "Thumb_Up":
        ("VOLUME UP", "volumeup"),

    "Thumb_Down":
        ("VOLUME DOWN", "volumedown"),

    "Victory":
        ("NEXT TRACK", "nexttrack"),

    "Closed_Fist":
        ("PREVIOUS TRACK", "prevtrack")
}


def perform_action(gesture):

    global last_action
    global last_action_time

    if gesture not in GESTURE_ACTIONS:

        return

    current_time = time.time()

    # Prevent continuous triggering
    if (
        gesture == last_action
        and
        current_time - last_action_time
        < ACTION_COOLDOWN
    ):

        return

    label, key = GESTURE_ACTIONS[gesture]

    pyautogui.press(key)

    print("Action:", label)

    last_action = gesture
    last_action_time = current_time


# ============================================================
# SWIPE DETECTION
# ============================================================

def detect_swipe(hand):

    global last_swipe_time

    current_time = time.time()

    wrist = hand[0]

    wrist_x = wrist.x

    # Save current wrist position
    wrist_history.append(
        (current_time, wrist_x)
    )

    if len(wrist_history) < 5:

        return None

    # Remove positions older than SWIPE_TIME
    while (
        wrist_history
        and
        current_time - wrist_history[0][0]
        > SWIPE_TIME
    ):

        wrist_history.popleft()

    if len(wrist_history) < 5:

        return None

    first_time, first_x = wrist_history[0]

    movement = wrist_x - first_x

    # Don't trigger another swipe immediately
    if (
        current_time - last_swipe_time
        < SWIPE_COOLDOWN
    ):

        return None

    # Swipe RIGHT
    if movement > SWIPE_DISTANCE:

        last_swipe_time = current_time

        wrist_history.clear()

        return "SWIPE RIGHT"

    # Swipe LEFT
    if movement < -SWIPE_DISTANCE:

        last_swipe_time = current_time

        wrist_history.clear()

        return "SWIPE LEFT"

    return None


# ============================================================
# SWIPE ACTION
# ============================================================

def perform_swipe(swipe):

    global last_action
    global last_action_time

    if swipe == "SWIPE RIGHT":

        pyautogui.press("nexttrack")

        print("Action: SWIPE RIGHT → NEXT TRACK")

        last_action = swipe
        last_action_time = time.time()

    elif swipe == "SWIPE LEFT":

        pyautogui.press("prevtrack")

        print("Action: SWIPE LEFT → PREVIOUS TRACK")

        last_action = swipe
        last_action_time = time.time()


# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(0)

camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)


if not camera.isOpened():

    print("ERROR: Could not open camera.")

    exit()


# ============================================================
# FPS
# ============================================================

previous_time = time.time()


# ============================================================
# MAIN LOOP
# ============================================================

with GestureRecognizer.create_from_options(
    options
) as recognizer:

    while True:

        success, frame = camera.read()

        if not success:

            print(
                "ERROR: Could not read camera frame."
            )

            break


        # Mirror camera
        frame = cv2.flip(
            frame,
            1
        )


        # BGR → RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )


        # Recognize
        result = recognizer.recognize(
            mp_image
        )


        gesture = "NO HAND"
        confidence = 0.0

        display_gesture = "NO HAND"


        # ====================================================
        # HAND DETECTED
        # ====================================================

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]


            # ------------------------------------------------
            # CHECK PINCH FIRST
            # ------------------------------------------------

            if is_pinch(hand):

                gesture_history.clear()

                display_gesture = "PINCH"

                # Pinch is a one-shot action
                current_time = time.time()

                if (
                    current_time - last_action_time
                    > ACTION_COOLDOWN
                ):

                    pyautogui.click()

                    print(
                        "Action: PINCH → LEFT CLICK"
                    )

                    last_action = "PINCH"

                    last_action_time = current_time


            else:

                # --------------------------------------------
                # NORMAL MEDIAPIPE GESTURE
                # --------------------------------------------

                if result.gestures:

                    if len(result.gestures[0]) > 0:

                        category = (
                            result.gestures[0][0]
                        )

                        gesture = (
                            category.category_name
                        )

                        confidence = (
                            category.score
                        )


                        if confidence >= MIN_CONFIDENCE:

                            stable_gesture = (
                                get_stable_gesture(
                                    gesture
                                )
                            )


                            if stable_gesture not in [
                                "Stabilizing...",
                                "UNKNOWN"
                            ]:

                                perform_action(
                                    stable_gesture
                                )


                            display_gesture = (
                                stable_gesture
                            )

                        else:

                            display_gesture = (
                                "LOW CONFIDENCE"
                            )

                else:

                    gesture_history.clear()

                    display_gesture = "UNKNOWN"


            # =================================================
            # SWIPE
            # =================================================

            swipe = detect_swipe(hand)


            if swipe:

                display_gesture = swipe

                perform_swipe(swipe)


            # =================================================
            # DRAW LANDMARKS
            # =================================================

            for point in hand:

                x = int(
                    point.x *
                    frame.shape[1]
                )

                y = int(
                    point.y *
                    frame.shape[0]
                )

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )


            # =================================================
            # DRAW CONNECTIONS
            # =================================================

            connections = (
                mp.tasks.vision
                .HandLandmarksConnections
                .HAND_CONNECTIONS
            )


            for connection in connections:

                start = hand[
                    connection.start
                ]

                end = hand[
                    connection.end
                ]


                p1 = (
                    int(
                        start.x *
                        frame.shape[1]
                    ),
                    int(
                        start.y *
                        frame.shape[0]
                    )
                )


                p2 = (
                    int(
                        end.x *
                        frame.shape[1]
                    ),
                    int(
                        end.y *
                        frame.shape[0]
                    )
                )


                cv2.line(
                    frame,
                    p1,
                    p2,
                    (0, 255, 0),
                    2
                )


        else:

            # No hand
            gesture_history.clear()

            wrist_history.clear()

            display_gesture = "NO HAND"


        # ====================================================
        # FPS
        # ====================================================

        current_time = time.time()

        fps = 1 / max(
            current_time - previous_time,
            0.001
        )

        previous_time = current_time


        # ====================================================
        # UI PANEL
        # ====================================================

        cv2.rectangle(
            frame,
            (15, 15),
            (520, 135),
            (0, 0, 0),
            -1
        )


        cv2.putText(
            frame,
            f"Gesture: {display_gesture}",
            (30, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            f"Confidence: {confidence:.2f}",
            (30, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (30, 112),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            "Q = Quit",
            (
                frame.shape[1] - 130,
                35
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


        # ====================================================
        # DISPLAY
        # ====================================================

        cv2.imshow(
            "Gesture Controller",
            frame
        )


        # ====================================================
        # QUIT
        # ====================================================

        if (
            cv2.waitKey(1) & 0xFF
            == ord("q")
        ):

            break


# ============================================================
# CLEANUP
# ============================================================

camera.release()

cv2.destroyAllWindows()