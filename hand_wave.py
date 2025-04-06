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
prev_angle = None
wave_counter = 0
threshold_angle = 130  # Threshold for angle in degrees

def calculate_angle(a, b, c):
    """
    Calculate the angle between three points using the dot product formula.
    a, b, c are points (x, y) in 2D.
    """
    # Vectors: AB and BC
    ab = [b[0] - a[0], b[1] - a[1]]
    bc = [c[0] - b[0], c[1] - b[1]]

    # Dot product
    dot_product = ab[0] * bc[0] + ab[1] * bc[1]
    # Magnitudes of the vectors
    magnitude_ab = math.sqrt(ab[0] ** 2 + ab[1] ** 2)
    magnitude_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)

    # Cosine of the angle
    cos_angle = dot_product / (magnitude_ab * magnitude_bc)
    
    # Angle in radians
    angle = math.acos(cos_angle)
    
    # Convert to degrees
    angle_deg = math.degrees(angle)
    return angle_deg

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
            # Draw hand landmarks on the frame
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Check if body landmarks (elbow) are detected
    if pose_results.pose_landmarks:
        landmarks = pose_results.pose_landmarks.landmark

        # Get shoulder, elbow, and wrist coordinates
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]

        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

        # Convert coordinates from normalized to pixel values
        frame_height, frame_width, _ = frame.shape
        left_shoulder = (int(left_shoulder.x * frame_width), int(left_shoulder.y * frame_height))
        left_elbow = (int(left_elbow.x * frame_width), int(left_elbow.y * frame_height))
        left_wrist = (int(left_wrist.x * frame_width), int(left_wrist.y * frame_height))

        right_shoulder = (int(right_shoulder.x * frame_width), int(right_shoulder.y * frame_height))
        right_elbow = (int(right_elbow.x * frame_width), int(right_elbow.y * frame_height))
        right_wrist = (int(right_wrist.x * frame_width), int(right_wrist.y * frame_height))

        # Calculate the angle for left arm (shoulder -> elbow -> wrist)
        left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

        # Calculate the angle for right arm (shoulder -> elbow -> wrist)
        right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

        # If the angle exceeds the threshold, it's a potential waving motion
        if left_angle > threshold_angle or right_angle > threshold_angle:
            wave_counter += 1  # Increment the wave counter for significant arm movement
            cv2.putText(frame, "Hand Waving Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Optional: Display the angle on the frame
        cv2.putText(frame, f"Left Arm Angle: {left_angle:.2f}°", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Right Arm Angle: {right_angle:.2f}°", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow("Hand Waving Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
