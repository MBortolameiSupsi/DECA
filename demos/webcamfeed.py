import cv2
import numpy as np

def main():
  
    # Capture video from the first camera device
    cap = cv2.VideoCapture(0)

    # Check if the video stream is opened successfully
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        exit()

    # Set the window name
    window_name = 'Webcam Augmented Feed'

    # Create a window to display the frames
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # Set the desired window size
    desired_width = 1920
    desired_height = 1200
    cv2.resizeWindow(window_name, desired_width, desired_height)
    # Loop to continuously get frames
    while True:
        # Read a frame
        ret, frame = cap.read()

        # If frame is read correctly, ret is True
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        # Process the frame
        processed_frame = process_frame(frame)

        # Display the resulting frame
        cv2.imshow(window_name, processed_frame)

        # Press 'q' on the keyboard to exit the loop
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Function to perform some manipulation on the frame
def process_frame(frame):
    # Example: Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Insert your manipulation here, like applying 3D landmarks using DECA
    # processed_frame = apply_deca_landmarks(gray_frame)
    return gray_frame  # return processed_frame when DECA landmarks are applied

if __name__ == '__main__':
    main()