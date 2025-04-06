import cv2
import mediapipe as mp
import numpy as np
import csv

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to load labeled data from CSV file
def load_labeled_data(filename):
    data = []
    labels = []
    with open(filename, mode='r') as f:
        reader = csv.reader(f)
        for row in reader:
            labels.append(row[0])  # First column is the label
            data.append(np.array(row[1:], dtype=np.float32))  # Rest are landmarks
    return np.array(data), np.array(labels)

# Function to classify posture using nearest neighbor approach
def classify_posture(landmarks, labeled_data, labels):
    input_data = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks]).flatten()
    distances = np.linalg.norm(labeled_data - input_data, axis=1)
    return labels[np.argmin(distances)]  # Return label of closest match

# Load labeled data from CSV file
csv_file = "pose_landmarks.csv"  # Replace with your CSV file path
try:
    labeled_data, labels = load_labeled_data(csv_file)
    print(f"Loaded {len(labels)} labeled postures from {csv_file}.")
except FileNotFoundError:
    print(f"Error: File '{csv_file}' not found.")
    exit()

# Start video capture
cap = cv2.VideoCapture(1)

print("Press 'q' to quit.")

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
        
        # Classify posture based on landmarks
        posture_label = classify_posture(results.pose_landmarks.landmark, labeled_data, labels)
        
        # Display detected posture on the frame
        cv2.putText(frame, f'Detected Posture: {posture_label}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame with posture detection
    cv2.imshow('Posture Detection', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
