import cv2
import numpy as np
import os


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    commonImageFromCamera1Path = default = os.path.join(
        script_dir, "./commonImageFromWebcam_640x360.jpg"
    )
    commonImageFromCamera2Path = default = os.path.join(
        script_dir, "./commonImageFromLucid_640x360.jpg"
    )

    intrinsicsCamera1Path = default = os.path.join(script_dir, "./calibration_data_webcam_side_640_2.npz")
    intrinsicsCamera2Path = default = os.path.join(script_dir, "./calibration_data_lucid3_640.npz")
    intrinsicsCamera1 = np.load(intrinsicsCamera1Path)
    intrinsicsCamera2 = np.load(intrinsicsCamera2Path)

    retCamera1, rvecsCamera1, tvecsCamera1 = getExtrinsics(
        commonImageFromCamera1Path, intrinsicsCamera1, 
    )
    rotationMatrixCamera1 = cv2.Rodrigues(rvecsCamera1)[0].T
    distFromCamera1 = np.linalg.norm(tvecsCamera1)/10 # in cm
    print(f"WEBCAM: ret {retCamera1}, \nrotationMatrix {rotationMatrixCamera1}, \ntVecs {tvecsCamera1}, \nDIST:{distFromCamera1:.01f}cm")
    
    retCamera2, rvecsCamera2, tvecsCamera2 = getExtrinsics(
    commonImageFromCamera2Path, intrinsicsCamera2, 
    )
    rotationMatrixCamera2 = cv2.Rodrigues(rvecsCamera2)[0].T
    distFromCamera2 = np.linalg.norm(tvecsCamera2)/10 # in cm
    print(f"LUCID: ret {retCamera2}, \nrotationMatrix {rotationMatrixCamera2}, \ntVecs {tvecsCamera2}, \nDIST:{distFromCamera2:.01f}cm")
    

def getExtrinsics(imagePath, intrinsics):
    rows = 6
    columns = 9
    # Termination criteria for the corner sub-pixel algorithm
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    # Define the size of chessboard squares in millimeters.
    square_size = (
        23  # Adjust the square size to the actual size of your chessboard squares.
    )
    # Prepare object points with actual square size.
    objp = objp * square_size

    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (rows, columns), None)

    if ret == True:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    else:
        print(f"ERROR, cannot find corners in {imagePath}")
        return None
    # Find the rotation and translation vectors.
    ret, rvecs, tvecs = cv2.solvePnP(objp, corners2, intrinsics["camera_matrix"], intrinsics["dist_coeffs"])
    return ret, rvecs, tvecs


if __name__ == "__main__":
    main()