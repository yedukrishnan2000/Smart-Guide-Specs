import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Hands and Pose (for arm and body landmarks)
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Start capturing video
cap = cv2.VideoCapture(1)

# Variables to track movement
prev_wrist_x = None
prev_wrist_y = None
prev_shoulder_x = None
prev_shoulder_y = None
wave_counter = 0
threshold = 0.1  # Threshold for movement detection (adjust as needed)

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for hand and body detection
    hand_results = hands.process(rgb_frame)
    pose_results = pose.process(rgb_frame)

    # Check if hands are detected
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            # Get wrist coordinates (landmark 0 for wrist)
            wrist_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
            wrist_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y

            # If previous wrist position exists, calculate the movement distance
            if prev_wrist_x is not None and prev_wrist_y is not None:
                distance = calculate_distance(prev_wrist_x, prev_wrist_y, wrist_x, wrist_y)

                # Check if the movement is significant
                if distance > threshold:
                    wave_counter += 1  # Increment the wave counter for significant movement

            # Update previous wrist position
            prev_wrist_x, prev_wrist_y = wrist_x, wrist_y

            # If enough waves are detected, classify as hand waving
            if wave_counter > 5:  # Adjust threshold based on your needs
                cv2.putText(frame, "Hand Waving Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                wave_counter = 0  # Reset the wave counter after detecting a wave

            # Draw hand landmarks on the frame
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Check if body landmarks (shoulder) are detected
    if pose_results.pose_landmarks:
        # Draw all body landmarks on the frame
        mp.solutions.drawing_utils.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Optional: You can extract specific body landmark positions if needed
        landmarks = pose_results.pose_landmarks.landmark
        for idx, landmark in enumerate(landmarks):
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])

            # Optional: Display the landmark number on the frame
            cv2.putText(frame, f'{idx}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Get shoulder coordinates (landmark 11 for left shoulder, 12 for right shoulder)
        left_shoulder_x = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x
        left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
        right_shoulder_x = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x
        right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y

        # If previous shoulder position exists, calculate the arm movement
        if prev_shoulder_x is not None and prev_shoulder_y is not None:
            left_shoulder_distance = calculate_distance(prev_shoulder_x, prev_shoulder_y, left_shoulder_x, left_shoulder_y)
            right_shoulder_distance = calculate_distance(prev_shoulder_x, prev_shoulder_y, right_shoulder_x, right_shoulder_y)

            # Check if the shoulder moved significantly
            if left_shoulder_distance > threshold or right_shoulder_distance > threshold:
                cv2.putText(frame, "Arm Moving", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Update previous shoulder position
        prev_shoulder_x, prev_shoulder_y = left_shoulder_x, left_shoulder_y

    # Display the frame
    cv2.imshow("Body Landmark Tracking", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
