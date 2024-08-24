import cv2
from ultralytics import YOLO
import time

# Load the YOLOv8 model (optional, as you are only doing edge detection here)
model = YOLO("yolov8n-seg.pt")

# Open the video file or webcam (use 0 for webcam)
video_path = "8350149-uhd_3840_2160_25fps.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    success, frame = cap.read()

    if success:
        # Perform Canny edge detection on grayscale version of the frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_frame, 100, 200)

        # Create a mask where the edges are white
        mask = edges != 0

        # Apply the mask on the original color frame
        color_edges = frame.copy()
        color_edges[mask] = [0, 255, 0]  # Green edges (you can change this color)

        # Display the frame with colored edges
        cv2.imshow("Color Edge Detection", color_edges)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
