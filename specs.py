import cv2
import mediapipe as mp
import math
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, db
import urllib.request
import numpy as np

# Initialize Firebase
cred = credentials.Certificate("specs.json")  # Replace with your Firebase credentials file
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://specs-94571-default-rtdb.firebaseio.com/'  # Replace with your Firebase Realtime Database URL
})

# Load YOLOv8 model
model = YOLO("yolov8s.pt")  # You can change the model if needed

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Camera stream URL
video_url = "http://192.168.1.45/640x480.mjpeg"

# Variables to track movement
prev_size = None
prev_x_left_hip = None
prev_x_right_hip = None
walking_direction = None
motion_threshold = 5
no_motion_threshold = 0.05
size_tolerance = 0.02
horizontal_tolerance = 5
left_hand_wave_angle_threshold = 100  # Angle threshold for hand waving
right_hand_wave_angle_threshold = 300

# Function to calculate the distance between two points
def calculate_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    angle = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    if angle < 0:
        angle += 360
    return angle

# Function to send data to Firebase
def send_to_firebase(hand_wave, walking_direction, detected_objects):
    ref = db.reference('results')  # Reference to the 'results' node in Firebase
    ref.set({
        'hand_wave': hand_wave,
        'walking_direction': walking_direction,
        'detected_objects': detected_objects
    })

# Open the IP camera stream
try:
    stream = urllib.request.urlopen(video_url)
    bytes_data = b""
    print("Successfully connected to the camera stream")
except Exception as e:
    print(f"Error: Unable to access video stream. {e}")
    exit()

while True:
    try:
        # Read from the IP camera stream
        bytes_data += stream.read(1024)
        a = bytes_data.find(b'\xff\xd8')  # JPEG start
        b = bytes_data.find(b'\xff\xd9')  # JPEG end
        
        if a != -1 and b != -1:
            jpg = bytes_data[a:b+2]
            bytes_data = bytes_data[b+2:]
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            
            if frame is None:
                print("Error: Could not decode frame")
                continue
                
            # Convert the frame to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Perform object detection using YOLO
            results = model(frame)  # You can also resize the frame before passing it

            # Process the frame for body landmarks detection
            pose_results = pose.process(rgb_frame)

            # Check if body landmarks are detected
            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark

                # Get relevant landmarks for detecting hand waving and body movements
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
                left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
                
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
                right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

                # Convert normalized coordinates to pixel coordinates
                frame_height, frame_width, _ = frame.shape
                left_shoulder = (int(left_shoulder.x * frame_width), int(left_shoulder.y * frame_height))
                left_elbow = (int(left_elbow.x * frame_width), int(left_elbow.y * frame_height))
                left_wrist = (int(left_wrist.x * frame_width), int(left_wrist.y * frame_height))
                
                right_shoulder = (int(right_shoulder.x * frame_width), int(right_shoulder.y * frame_height))
                right_elbow = (int(right_elbow.x * frame_width), int(right_elbow.y * frame_height))
                right_wrist = (int(right_wrist.x * frame_width), int(right_wrist.y * frame_height))

                left_hip = (int(left_hip.x * frame_width), int(left_hip.y * frame_height))
                right_hip = (int(right_hip.x * frame_width), int(right_hip.y * frame_height))

                # Calculate the angle for hand waving (left hand)
                left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

                # Detect hand waving, ensuring the wrist is above the elbow
                hand_wave = "No Wave"
                if left_wrist[1] < left_elbow[1] and left_angle > left_hand_wave_angle_threshold:
                    hand_wave = "Hand Waving"
                elif right_wrist[1] < right_elbow[1] and right_angle > right_hand_wave_angle_threshold:
                    hand_wave = "Hand Waving"
                
                # Calculate body size by measuring the distance between hips
                hip_distance = calculate_distance(left_hip, right_hip)
                body_size = hip_distance

                # Detect if the size increases or decreases
                if prev_size is not None:
                    size_change = body_size - prev_size
                    if abs(size_change) < no_motion_threshold:
                        walking_direction = "No Motion"
                    elif body_size > prev_size + motion_threshold:
                        walking_direction = "Walking Towards"
                    elif body_size < prev_size - motion_threshold:
                        walking_direction = "Walking Away"
                else:
                    walking_direction = "No Motion"

                # Track horizontal movement (left/right) only if no forward/away motion is detected
                if walking_direction == "No Motion":
                    if prev_x_left_hip is not None and prev_x_right_hip is not None:
                        # Check horizontal position tolerance
                        if abs(left_hip[0] - prev_x_left_hip) < horizontal_tolerance and abs(right_hip[0] - prev_x_right_hip) < horizontal_tolerance:
                            walking_direction = "No Motion"
                        elif left_hip[0] > prev_x_left_hip or right_hip[0] > prev_x_right_hip:
                            walking_direction = "Walking Left"
                        elif left_hip[0] < prev_x_left_hip or right_hip[0] < prev_x_right_hip:
                            walking_direction = "Walking Right"
                
                # Update previous values
                prev_size = body_size
                prev_x_left_hip = left_hip[0]
                prev_x_right_hip = right_hip[0]

                # Collect detected objects from YOLO results
                detected_objects = []
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        label_idx = int(box.cls[0].cpu().numpy())
                        label = result.names[label_idx]
                        detected_objects.append(label)

                # Send results to Firebase
                send_to_firebase(hand_wave, walking_direction, detected_objects)

                # Show the results on the frame
                cv2.putText(frame, f"Wave: {hand_wave}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Movement: {walking_direction}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display angles on screen
                cv2.putText(frame, f"Left Angle: {left_angle:.2f}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Right Angle: {right_angle:.2f}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Optionally: Draw landmarks and connections
                mp.solutions.drawing_utils.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Loop through the YOLOv8 detections and draw bounding boxes
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    label_idx = int(box.cls[0].cpu().numpy())
                    label = result.names[label_idx]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green bounding box
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Display the frame with both detections
            cv2.imshow("Combined Detection (Pose + YOLO)", frame)

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"Error during processing: {e}")
        continue

# Clean up
stream.close()
cv2.destroyAllWindows()