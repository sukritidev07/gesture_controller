import cv2
import mediapipe as mp

# Create the Hand Landmarker
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

# Open the camera
camera = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:

    while True:
        success, frame = camera.read()

        if not success:
            print("Could not access camera")
            break

        # Mirror the camera
        frame = cv2.flip(frame, 1)

        # Convert OpenCV BGR → RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert frame to MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Detect hands
        result = landmarker.detect(mp_image)

        # Draw hand landmarks
        if result.hand_landmarks:

            for hand in result.hand_landmarks:

                for landmark in hand:

                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])

                    cv2.circle(
                        frame,
                        (x, y),
                        5,
                        (0, 255, 0),
                        -1
                    )

                # Draw connections
                connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

                for connection in connections:

                    start = hand[connection.start]
                    end = hand[connection.end]

                    start_point = (
                        int(start.x * frame.shape[1]),
                        int(start.y * frame.shape[0])
                    )

                    end_point = (
                        int(end.x * frame.shape[1]),
                        int(end.y * frame.shape[0])
                    )

                    cv2.line(
                        frame,
                        start_point,
                        end_point,
                        (0, 255, 0),
                        2
                    )

        cv2.imshow("Gesture Controller", frame)

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()