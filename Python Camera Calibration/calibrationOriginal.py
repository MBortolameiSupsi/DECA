import numpy as np
import cv2 as cv
import glob

# termination criteria for the corner sub-pixel algorithm
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# assuming a 7x6 chessboard pattern
objp = np.zeros((7*6, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d points in real-world space
imgpoints = []  # 2d points in image plane.

# Change the directory and image format according to your setup
images = glob.glob('./images_original/*.jpg')
print(f"images found {len(images)}")

# Define the size of chessboard squares
#square_size = 23.0  # in millimeters
square_size = 1  # in millimeters

# Prepare object points based on the detected chessboard size
objp = np.zeros((7*6, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2) * square_size
print("objp is ",objp)

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv.findChessboardCorners(gray, (7, 6), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        print(f"corners {len(corners)}")  # This should be inside the if block
        objpoints.append(objp)
        # refine the corner positions
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        #breakpoint()

        # Draw and display the corners
        img = cv.drawChessboardCorners(img, (7, 6), corners2, ret)  # Size should match (9, 6)
        cv.imshow('img', img)
        cv.waitKey(500)

cv.destroyAllWindows()

# Calibration
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("dist is ",dist, "mtx is ",mtx)
#dist = np.zeros((1, 5), np.float32)
# Undistort an image
img = cv.imread('./images_original/left12.jpg')
h, w = img.shape[:2]
#w, h = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
#dst = cv.undistort(img, mtx, dist)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresultOriginal.png', dst)

# Calculate the total error of the calibration
total_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    total_error += error

print("Total error: ", total_error / len(objpoints))

# Save the camera matrix and the distortion coefficients to be used for future images
np.savez('calibration_data.npz', camera_matrix=mtx, dist_coeffs=dist, rvecs=rvecs, tvecs=tvecs)