import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Pose Estimation
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define activity recognition function
def recognize_activity(landmarks):
    # Extract key landmarks for activity detection
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    left_foot = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]
    right_foot = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]
    
    # Example logic for activity detection (simplified)
    if left_wrist.y < right_wrist.y and abs(left_wrist.x - right_wrist.x) > 0.2:
        return "Wave"
    elif abs(left_foot.y - right_foot.y) > 0.3:
        return "Jump"
    elif abs(left_foot.x - right_foot.x) < 0.1:
        return "Walk/Run"
    elif left_wrist.y > left_foot.y and right_wrist.y > right_foot.y:
        return "Sit"
    else:
        return "Unknown"

# Start video capture
cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB for Mediapipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with Mediapipe Pose
    results = pose.process(rgb_frame)
    
    if results.pose_landmarks:
        # Draw pose landmarks on the frame
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Recognize activity based on pose landmarks
        activity = recognize_activity(results.pose_landmarks.landmark)
        
        # Display the recognized activity on the frame
        cv2.putText(frame, f'Activity: {activity}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame with activity recognition
    cv2.imshow('Activity Tracking', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
