"""
camera.py  —  Webcam Test Utility
Tests that the webcam is accessible and displays a preview.
Press Q to quit.
"""

import cv2

# Try index 0 first; change to 1 if using external webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
else:
    print("Webcam is working. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        cv2.imshow("Webcam Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
