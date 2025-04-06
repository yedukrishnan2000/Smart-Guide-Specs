import cv2
from ultralytics import YOLO

# Load YOLOv8 model (YOLOv8 is available in ultralytics package)
model = YOLO("yolov8s.pt")  # You can change the model type based on your need (e.g., yolov8s.pt, yolov8m.pt, yolov8l.pt)

# Open the webcam (0 is typically the default webcam)
cap = cv2.VideoCapture(1)  # Change this if you want to capture from a different camera or use a video file

while cap.isOpened():
    ret, frame = cap.read()  # Read each frame from the webcam
    if not ret:
        print("Failed to grab frame")
        break

    # Perform inference on the current frame
    results = model(frame)

    # Iterate through each result to manually draw bounding boxes and labels
    for result in results:
        # Get the bounding boxes and labels
        boxes = result.boxes
        for box in boxes:
            # Get the coordinates of the bounding box
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            # Get the class label as an integer
            label_idx = int(box.cls[0].cpu().numpy())  # Convert to int
            label = result.names[label_idx]  # Get the label from names dictionary
            # Draw the bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green bounding box
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the frame with bounding boxes and labels
    cv2.imshow("YOLOv8 Object Detection", frame)

    # Optionally, add a condition to stop the video capture if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
