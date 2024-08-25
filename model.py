import cv2
from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")

image_path = "Capture3.PNG" 
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not load image.")
else:
    # Converting  the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray_image, 100, 200)

    # Creating a mask where the edges are white
    mask = edges != 0

    # Applying the mask on the original color image
    color_edges = image.copy()
    color_edges[mask] = [0, 255, 0]  # Green edges

    output_path = "color_edges_output.png"  # Path
    cv2.imwrite(output_path, color_edges)


    cv2.imshow("Color Edge Detection", color_edges)

    
cv2.waitKey(0)
cv2.destroyAllWindows()
