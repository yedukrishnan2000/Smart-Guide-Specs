import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Start capturing video
cap = cv2.VideoCapture(1)

# Variables to track movement
prev_size = None
prev_x_left_hip = None
prev_x_right_hip = None
walking_direction = None  # Will be used to detect walking directions
motion_threshold = 5  # Minimum size change to consider as motion
no_motion_threshold = 0.05  # If the size change is less than this, it's no motion
size_tolerance = 0.02  # Tolerance for size change (to consider no motion)
horizontal_tolerance = 5  # Tolerance for horizontal position change (x-coordinate)

# Function to calculate the distance between two points
def calculate_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for body landmarks detection
    pose_results = pose.process(rgb_frame)

    # Check if body landmarks are detected
    if pose_results.pose_landmarks:
        landmarks = pose_results.pose_landmarks.landmark

        # Get hip, knee, and ankle landmarks (using index numbers from MediaPipe Pose)
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
        
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

        # Convert normalized coordinates to pixel coordinates
        frame_height, frame_width, _ = frame.shape
        left_hip = (int(left_hip.x * frame_width), int(left_hip.y * frame_height))
        left_knee = (int(left_knee.x * frame_width), int(left_knee.y * frame_height))
        left_ankle = (int(left_ankle.x * frame_width), int(left_ankle.y * frame_height))

        right_hip = (int(right_hip.x * frame_width), int(right_hip.y * frame_height))
        right_knee = (int(right_knee.x * frame_width), int(right_knee.y * frame_height))
        right_ankle = (int(right_ankle.x * frame_width), int(right_ankle.y * frame_height))

        # Calculate the size by measuring the distance between key landmarks
        hip_distance = calculate_distance(left_hip, right_hip)
        body_size = hip_distance

        # Detect if the size increases or decreases
        if prev_size is not None:
            size_change = body_size - prev_size
            if abs(size_change) < no_motion_threshold:
                walking_direction = "No Motion"  # No significant change in size
            elif body_size > prev_size + motion_threshold:
                walking_direction = "Walking Towards"  # Size is increasing
            elif body_size < prev_size - motion_threshold:
                walking_direction = "Walking Away"  # Size is decreasing
        else:
            walking_direction = "No Motion"  # Initial frame where no motion is detected yet

        # Track horizontal movement (left/right)
        # We will track the x-coordinate of the left and right hips
        if prev_x_left_hip is not None and prev_x_right_hip is not None:
            # Check horizontal position tolerance
            if abs(left_hip[0] - prev_x_left_hip) < horizontal_tolerance and abs(right_hip[0] - prev_x_right_hip) < horizontal_tolerance:
                walking_direction = "No Motion"  # No significant change in horizontal position

            # Detect left/right walking based on x-coordinate changes
            if left_hip[0] > prev_x_left_hip:
                walking_direction = "Walking Left"
            elif left_hip[0] < prev_x_left_hip:
                walking_direction = "Walking Right"
            if right_hip[0] > prev_x_right_hip:
                walking_direction = "Walking Left"
            elif right_hip[0] < prev_x_right_hip:
                walking_direction = "Walking Right"

        # Update previous size and x-coordinates
        prev_size = body_size
        prev_x_left_hip = left_hip[0]
        prev_x_right_hip = right_hip[0]

        # Show walking direction on the frame
        cv2.putText(frame, walking_direction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Optionally: Draw body landmarks on the frame
        mp.solutions.drawing_utils.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display the frame
    cv2.imshow("Walking Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
