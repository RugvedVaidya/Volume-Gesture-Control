import cv2
import mediapipe as mp
import math
import pyautogui

# Initialize hand tracking
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize PyAutoGUI
pyautogui.FAILSAFE = False

# Set up camera
cap = cv2.VideoCapture(0)
width, height = cap.get(3), cap.get(4)

# Define volume control parameters
volume_range = (0, 100)
min_distance = 50

def calculate_distance(hand_landmarks):
    x1, y1 = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x, hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
    x2, y2 = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x, hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2) * width
    return distance

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame
        results = hands.process(rgb_frame)

        # Check if hand landmarks are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Calculate distance between thumb tip and index finger tip
                distance = calculate_distance(hand_landmarks)

                # Map distance to volume range
                volume = int((distance / min_distance) * (volume_range[1] - volume_range[0]) + volume_range[0])
                volume = max(min(volume, volume_range[1]), volume_range[0])

                # Set system volume using PyAutoGUI
                pyautogui.press('volumedown') if volume < 50 else pyautogui.press('volumeup')

                # Draw hand landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display the frame
        cv2.imshow('Gesture Volume Control', frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture object
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
