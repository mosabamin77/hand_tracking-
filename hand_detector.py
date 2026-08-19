import cv2
import mediapipe as mp
import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==========================================
# SETTINGS
# ==========================================

MODEL_PATH = "hand_landmarker.task"

PEACE_IMAGE = "gestures\peace.jpg"
TWO_HANDS_IMAGE = "gestures\hands up.jpg"
SHUT_UP_IMAGE = "gestures\shut up.jpg"


# ==========================================
# CREATE MEDIAPIPE HAND DETECTOR
# ==========================================

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = vision.HandLandmarker.create_from_options(options)


# ==========================================
# LOAD IMAGES
# ==========================================

peace_image = cv2.imread(PEACE_IMAGE)
two_hands_image = cv2.imread(TWO_HANDS_IMAGE)
shut_up_image = cv2.imread(SHUT_UP_IMAGE)

if peace_image is None:
    print("ERROR: Could not find peace.jpg")

if two_hands_image is None:
    print("ERROR: Could not find hands up.jpg")

if shut_up_image is None:
    print("ERROR: Could not find shut up.jpg")


# ==========================================
# FUNCTION: CHECK IF FINGER IS UP
# ==========================================

def finger_is_up(hand, finger_tip, finger_pip):

    return hand[finger_tip].y < hand[finger_pip].y


# ==========================================
# FUNCTION: COUNT FINGERS
# ==========================================

def count_fingers(hand):

    fingers = 0

    # Index finger
    if finger_is_up(hand, 8, 6):
        fingers += 1

    # Middle finger
    if finger_is_up(hand, 12, 10):
        fingers += 1

    # Ring finger
    if finger_is_up(hand, 16, 14):
        fingers += 1

    # Pinky
    if finger_is_up(hand, 20, 18):
        fingers += 1

    return fingers


# ==========================================
# FUNCTION: DETECT PEACE SIGN
# ==========================================

def is_peace(hand):

    index_up = finger_is_up(hand, 8, 6)
    middle_up = finger_is_up(hand, 12, 10)
    ring_down = not finger_is_up(hand, 16, 14)
    pinky_down = not finger_is_up(hand, 20, 18)

    return (
        index_up
        and middle_up
        and ring_down
        and pinky_down
    )
def is_shut_up(hand):

    index_up = finger_is_up(hand, 8, 6)
    middle_down = not finger_is_up(hand, 12, 10)
    ring_down = not finger_is_up(hand, 16, 14)
    pinky_down = not finger_is_up(hand, 20, 18)

    return (
        index_up
        and middle_down
        and ring_down
        and pinky_down
    )    


# ==========================================
# FUNCTION: DRAW HAND
# ==========================================

def draw_hand(frame, hand):

    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),

        (0, 5), (5, 6), (6, 7), (7, 8),

        (0, 9), (9, 10), (10, 11), (11, 12),

        (0, 13), (13, 14), (14, 15), (15, 16),

        (0, 17), (17, 18), (18, 19), (19, 20),

        (5, 9),
        (9, 13),
        (13, 17)
    ]

    # Draw landmarks

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

    for start, end in connections:

        x1 = int(hand[start].x * frame.shape[1])
        y1 = int(hand[start].y * frame.shape[0])

        x2 = int(hand[end].x * frame.shape[1])
        y2 = int(hand[end].y * frame.shape[0])

        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


# ==========================================
# FUNCTION: RESIZE IMAGE
# ==========================================

def resize_image(image, width, height):

    return cv2.resize(
        image,
        (width, height)
    )


# ==========================================
# OPEN CAMERA
# ==========================================

# CAP_DSHOW helps avoid Windows MSMF camera problems

cap = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)

if not cap.isOpened():

    print("Could not access the camera.")
    exit()


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    success, frame = cap.read()

    if not success:

        print("Could not read camera frame.")
        break


    # Mirror camera

    frame = cv2.flip(
        frame,
        1
    )


    # ======================================
    # CONVERT CAMERA FOR MEDIAPIPE
    # ======================================

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # ======================================
    # DETECT HANDS
    # ======================================

    result = detector.detect(
        mp_image
    )


    gesture = "No gesture"

    selected_image = None


    # ======================================
    # CHECK HANDS
    # ======================================

    if result.hand_landmarks:

        number_of_hands = len(
            result.hand_landmarks
        )


        # Draw every detected hand

        for hand in result.hand_landmarks:

            draw_hand(
                frame,
                hand
            )


        # ==================================
        # TWO HANDS UP
        # ==================================

        if number_of_hands == 2:

            fingers_hand_1 = count_fingers(
                result.hand_landmarks[0]
            )

            fingers_hand_2 = count_fingers(
                result.hand_landmarks[1]
            )


            # Both hands must have at least
            # 3 fingers raised

            if (
                fingers_hand_1 >= 3
                and fingers_hand_2 >= 3
            ):

                gesture = "HANDS UP"

                selected_image = two_hands_image


        # ==================================
        # PEACE SIGN
        # ==================================

        elif number_of_hands == 1:

            hand = result.hand_landmarks[0]

            if is_peace(hand):

                gesture = "PEACE"

                selected_image = peace_image
            elif is_shut_up(hand):

                gesture = "SHH!"
                
                selected_image = shut_up_image 


    # ======================================
    # DISPLAY GESTURE TEXT
    # ======================================

    cv2.putText(
        frame,
        gesture,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )


    # ======================================
    # PREPARE IMAGE PANEL
    # ======================================

    camera_width = 640
    camera_height = 480

    frame = resize_image(
        frame,
        camera_width,
        camera_height
    )


    # Create blank image panel

    image_panel = 255 * (
        __import__("numpy").ones(
            (
                camera_height,
                camera_width,
                3
            ),
            dtype="uint8"
        )
    )


    # ======================================
    # SHOW SELECTED IMAGE
    # ======================================

    if selected_image is not None:

        image_panel = resize_image(
            selected_image,
            camera_width,
            camera_height
        )

    else:

        cv2.putText(
            image_panel,
            "Make a gesture",
            (150, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 0),
            3
        )


    # ======================================
    # COMBINE CAMERA + IMAGE
    # ======================================

    combined = cv2.hconcat(
        [
            frame,
            image_panel
        ]
    )


    # ======================================
    # SHOW WINDOW
    # ======================================

    cv2.imshow(
        "Hand Gesture Controller",
        combined
    )


    # ======================================
    # QUIT
    # ======================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ==========================================
# CLEANUP
# ==========================================

cap.release()

cv2.destroyAllWindows()