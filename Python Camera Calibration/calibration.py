import numpy as np
import cv2 as cv
import glob
import argparse
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# Set up argument parsing
parser = argparse.ArgumentParser(description='Calibrate camera based on input images.')
parser.add_argument('-imagesFolder', '--imagesFolder', type=str, required=True, help='Directory containing JPEG files to use for calibration.')
parser.add_argument('-outputPath', '--outputPath', type=str, required=True, help='Full output path')
parser.add_argument('-outputName', '--outputName', type=str, required=True, help='Output filename without extension')

args = parser.parse_args()
imagesFolder = args.imagesFolder
outputPath = args.outputPath
outputName = args.outputName

# Termination criteria for the corner sub-pixel algorithm
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
rows = 6
columns = 9
# Define the size of chessboard squares in millimeters.
square_size = 23  # Adjust the square size to the actual size of your chessboard squares.

# Use this, since it's last
# imagesFolder = './images_lucidcamera3_640'
# imageToUndistort = '/img (15)_640x480.jpg'
# calibrationResultImage = 'calibresult_img (15).png'
# calibrationData = 'calibration_data_lucid3_640'

objp = np.zeros((rows*columns, 3), np.float32)
objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
# Prepare object points with actual square size.
objp = objp * square_size

# Arrays to store object points and image points from all images.
# used for error calculation
objpoints = []  # 3d points in real-world space
imgpoints = []  # 2d points in image plane.

# Gather all images in folder
images = glob.glob(imagesFolder+'/*.jpg')
print(f"Images found: {len(images)}")



#print("objp is ", objp)
# breakpoint()
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chessboard corners.
    ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)
    print(f"Image current: {fname} - corners {corners} ret {ret}")

    # If found, add object points, image points (after refining them).
    if ret:
        print(f"Corners found: {len(corners)}")
        objpoints.append(objp)
        
        # Refine the corner positions.
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners.
        img = cv.drawChessboardCorners(img, (rows, columns), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)
        # breakpoint()

cv.destroyAllWindows()

# Perform the camera calibration.
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("Distortion coefficients:", dist, "\nCamera matrix:", mtx)
print("tvecs:", tvecs)

for t in tvecs:
    print(f"distance is {np.linalg.norm(t)}")

# # Undistort an example image.
# # Make sure to use an image that exists in your directory.
# img = cv.imread(imagesFolder+imageToUndistort)
# h, w = img.shape[:2]
# newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# # Undistort the image.
# dst = cv.undistort(img, mtx, dist, None, newcameramtx)

# # Crop the image based on the ROI.
# x, y, w, h = roi
# # disable cropping
# #dst = dst[y:y+h, x:x+w]
# cv.imwrite(calibrationResultImage, dst)

# Calculate the total error of the calibration.
total_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    total_error += error

print("Total error:", total_error / len(objpoints))

# Save the camera matrix and the distortion coefficients to be used for future images.
np.savez(os.path.join(outputPath, f"{outputName}.npz"), camera_matrix=mtx, dist_coeffs=dist, rvecs=rvecs, tvecs=tvecs)
# Save also local copy for backup
np.savez( os.path.join(script_dir, "results", f"{outputName}.npz") , camera_matrix=mtx, dist_coeffs=dist, rvecs=rvecs, tvecs=tvecs)