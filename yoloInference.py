import cv2
import torch
from ultralytics import YOLO
import numpy as np
from sort import Sort
import psutil
import time

# Load the pre-trained YOLOv8n model
model = YOLO('yolov8n.pt')

# Initialize the SORT tracker
tracker = Sort(max_age=30)

# Initialize entry and exit counters
enter_count = 0
leave_count = 0

# Initialize a dictionary to track the state of each object
object_states = {}

# Define the polygon for the region of interest (ROI)
polygon = np.array([[97, 382], [237, 339], [248, 386], [105, 425]], np.int32)


# Function to check if a point is inside the polygon
def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0


# Define the function to apply the model on each frame
def detect_objects(frame, model, threshold=0.4):
    results = model(frame)[0]

    pred_boxes = results.boxes.xyxy.cpu().numpy()  # Bounding boxes
    pred_scores = results.boxes.conf.cpu().numpy()  # Confidence scores
    pred_labels = results.boxes.cls.cpu().numpy()  # Class labels

    keep = pred_scores > threshold
    pred_boxes = pred_boxes[keep]
    pred_scores = pred_scores[keep]
    pred_labels = pred_labels[keep]

    return pred_boxes, pred_scores, pred_labels


def draw_boxes(frame, tracked_objects, polygon, object_states):
    global enter_count, leave_count

    cv2.polylines(frame, [polygon], isClosed=True, color=(255, 0, 0), thickness=3)

    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = map(int, obj)
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

        # Check if the object is inside the polygon
        inside_polygon = point_in_polygon((cx, y2), polygon)

        # If the object is already being tracked
        if obj_id in object_states:
            # Get the previous state
            prev_inside_polygon = object_states[obj_id]

            # Update counts based on transition
            if prev_inside_polygon and not inside_polygon:
                leave_count += 1
            elif not prev_inside_polygon and inside_polygon:
                enter_count += 1

            # Update the state
            object_states[obj_id] = inside_polygon
        else:
            # If the object is new, initialize its state
            object_states[obj_id] = inside_polygon

        # Draw the rectangle and the object ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_text = f'ID: {obj_id}'
        cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.circle(frame, (cx, y2), 5, (0, 0, 255), -1)

    # Display enter and leave counts
    cv2.putText(frame, f'Enter: {enter_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    cv2.putText(frame, f'Leave: {leave_count}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

    return frame


# Open the video file
video_path = 'Input_video2.mp4'
output_path = 'yoloOutput.mp4'

cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, 1, (int(cap.get(3)), int(cap.get(4))))

# Initialize performance tracking
start_time = time.time()
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Measure start time for processing
    frame_start_time = time.time()

    # Detect objects
    boxes, scores, labels = detect_objects(frame, model)
    detections = np.hstack((boxes, scores[:, np.newaxis]))
    tracked_objects = tracker.update(detections)
    frame = draw_boxes(frame, tracked_objects, polygon, object_states)

    # Measure end time for processing
    frame_end_time = time.time()
    processing_time = frame_end_time - frame_start_time

    # Show the frame in a window
    cv2.imshow('Frame', frame)

    # Write the frame to the output video
    out.write(frame)

    # Display processing time and CPU usage
    print(f"Processing Time: {processing_time:.4f} seconds")
    cpu_usage = psutil.cpu_percent(interval=1)
    print(f"CPU Usage: {cpu_usage}%")

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

# Calculate and display overall performance metrics
end_time = time.time()
total_time = end_time - start_time
fps_calculated = frame_count / total_time
print(f"Overall FPS: {fps_calculated:.2f}")

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
