import numpy as np
import cv2 as cv
import glob

# Termination criteria for the corner sub-pixel algorithm
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
rows = 6
columns = 9

imagesFolder = './images_original'
imageToUndistort = '/left14.jpg'
calibrationResultImage = 'calibresult_original.png'
calibrationData = 'calibration_data_original'


imagesFolder = './images_webcam'
imageToUndistort = '/img (14).jpg'
calibrationResultImage = 'calibresult_webcam.png'
calibrationData = 'calibration_data_webcam'

imagesFolder = './images_webcam_cut'
imageToUndistort = '/img (11).jpg'
calibrationResultImage = 'calibresult_webcam_cut.png'
calibrationData = 'calibration_data_webcam_cut'

imagesFolder = './images_webcam_closer'
imageToUndistort = '/img (15).jpg'
calibrationResultImage = 'calibresult_webcam_closer.png'
calibrationData = 'calibration_data_webcam_closer'

imagesFolder = './images_iphone_back'
imageToUndistort = '/img (16).jpg'
calibrationResultImage = 'calibresult_iphone.png'
calibrationData = 'calibration_data_iphone_back'


imagesFolder = './images_iphone_back_2'
imageToUndistort = '/img (14).jpg'
calibrationResultImage = 'calibresult_iphone2.png'
calibrationData = 'calibration_data_iphone_back_2'

# Use this, since it's last
imagesFolder = './images_webcam_side'
imageToUndistort = '/img (6).jpg'
calibrationResultImage = 'calibresult_webcam_side.png'
calibrationData = 'calibration_data_webcam_side'

objp = np.zeros((rows*columns, 3), np.float32)
objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)

# Arrays to store object points and image points from all images.
objpoints = []  # 3d points in real-world space
imgpoints = []  # 2d points in image plane.

# Change the directory and image format according to your setup.
images = glob.glob(imagesFolder+'/*.jpg')
print(f"Images found: {len(images)}")

# Define the size of chessboard squares in millimeters.
square_size = 23  # Adjust the square size to the actual size of your chessboard squares.

# Prepare object points with actual square size.
objp = objp * square_size
#print("objp is ", objp)

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

cv.destroyAllWindows()

# Perform the camera calibration.
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("Distortion coefficients:", dist, "\nCamera matrix:", mtx)
print("tvecs:", tvecs)

# Undistort an example image.
# Make sure to use an image that exists in your directory.
img = cv.imread(imagesFolder+imageToUndistort)
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# Undistort the image.
dst = cv.undistort(img, mtx, dist, None, newcameramtx)

# Crop the image based on the ROI.
x, y, w, h = roi
# disable cropping
#dst = dst[y:y+h, x:x+w]
cv.imwrite(calibrationResultImage, dst)

# Calculate the total error of the calibration.
total_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    total_error += error

print("Total error:", total_error / len(objpoints))

# Save the camera matrix and the distortion coefficients to be used for future images.
np.savez(calibrationData+'.npz', camera_matrix=mtx, dist_coeffs=dist, rvecs=rvecs, tvecs=tvecs)