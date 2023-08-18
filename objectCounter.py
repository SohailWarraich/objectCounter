import cv2
import numpy as np

# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture('coin.mp4')

# Check if camera opened successfully
if not cap.isOpened():
    print("Error opening video file")

# Read until video is completed
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        # Define the region of interest (ROI)
        # roi = frame[y:y+h, x:x+w]
        roi = frame[177:501, 788:1269]  # Adjust the coordinates as needed
  
        # Draw a rectangle around the ROI
        cv2.rectangle(frame, (788, 177), (1269, 501), (0, 255, 0), 2)

        # Convert ROI to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to the ROI
        blur = cv2.GaussianBlur(gray, (11, 11), 0)

        # Apply Canny edge detection to the blurred ROI
        canny = cv2.Canny(blur, 30, 150)

        # Dilate the edges to close gaps
        dilated = cv2.dilate(canny, None, iterations=2)

        # Find contours in the dilated image
        contours, _ = cv2.findContours(
            dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on the original frame
        cv2.drawContours(roi, contours, -1, (0, 255, 0), 2)
        num_objects = len(contours)
        # Put text on the frame displaying the number of detected objects
        text = "Object detected: " + str(num_objects)
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the resulting frame with the drawn rectangle and contours
        cv2.imshow('Frame', frame)

        # Print the number of objects detected
        # print("Objects in the ROI: ", len(contours))

        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF in [ord('q'), ord('Q')]:
            break
    else:
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()