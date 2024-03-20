# from skimage.transform.hough_transform import label_distant_points
from operator import matmul
import cv2
import numpy as np
import os
import time

script_dir = os.path.dirname(os.path.abspath(__file__))


def main():
    commonImageFromCamera1Path = default = os.path.join(
        script_dir, "./commonImageWebcam_640x360.jpg"
    )
    commonImageFromCamera2Path = default = os.path.join(
        script_dir, "./commonImageLucid_640x480.jpg"
    )

    intrinsicsCamera1Path = default = os.path.join(
        script_dir, "./calibration_data_webcam_new.npz"
    )
    intrinsicsCamera2Path = default = os.path.join(
        script_dir, "./calibration_data_lucid_new.npz"
    )
    intrinsicsCamera1 = np.load(intrinsicsCamera1Path)
    intrinsicsCamera2 = np.load(intrinsicsCamera2Path)

    retCamera1, rvecsCamera1, tvecsCamera1, objPointsCamera1, imgPointsCamera1 = getExtrinsics(
        commonImageFromCamera1Path,
        intrinsicsCamera1,
    )
    rotationMatrixCamera1 = cv2.Rodrigues(rvecsCamera1)[0].T
    distFromCamera1 = np.linalg.norm(tvecsCamera1) / 10  # in cm
    print(
        f"WEBCAM: ret {retCamera1}, \nrotationMatrix {rotationMatrixCamera1}, \ntVecs {tvecsCamera1}, \nDIST:{distFromCamera1:.01f}cm"
    )

    retCamera2, rvecsCamera2, tvecsCamera2, objPointsCamera2, imgPointsCamera2 = getExtrinsics(
        commonImageFromCamera2Path,
        intrinsicsCamera2,
    )
    rotationMatrixCamera2 = cv2.Rodrigues(rvecsCamera2)[0].T
    distFromCamera2 = np.linalg.norm(tvecsCamera2) / 10  # in cm
    print(
        f"LUCID: ret {retCamera2}, \nrotationMatrix {rotationMatrixCamera2}, \ntVecs {tvecsCamera2}, \nDIST:{distFromCamera2:.01f}cm"
    

    reverseProjection(
        objPointsCamera1, 
        rvecsCamera1, 
        tvecsCamera1, 
        rvecsCamera2, 
        tvecsCamera2, 
        intrinsicsCamera2, 
        commonImageFromCamera2Path,
        imgPointsCamera2)
   

def reverseProjection(objp, rvecsCamera1, tvecsCamera1, rvecsCamera2, tvecsCamera2, intrinsicsCamera2, commonImageFromCamera2Path, imgPointsCamera2):
    # Step 1: Transform object points to Camera 1's coordinate system
    R_camera1, _ = cv2.Rodrigues(rvecsCamera1)
    objPoints_camera1 = np.dot(R_camera1, objp.T) + tvecsCamera1

    # Step 2: Compose transformations from Camera 1 to Camera 2

    # Invert Camera 1's transformation
    R_camera1_inv = R_camera1.T
    t_camera1_inv = -np.dot(R_camera1_inv, tvecsCamera1)

    # Apply Camera 2's transformation
    R_camera2, _ = cv2.Rodrigues(rvecsCamera2)
    t_camera2 = tvecsCamera2

    # Now, compose the transformations to map the points to Camera 2's coordinate system
    R_composed = np.dot(R_camera2, R_camera1_inv)
    t_composed = np.dot(R_camera2, t_camera1_inv) + t_camera2

    # Step 3: Reproject the points onto Camera 2's image plane
    imgpoints_from_1_to_2, _ = cv2.projectPoints(
        objPoints_camera1.T,  # Transpose back to the shape (N, 3)
        R_composed,
        t_composed,
        intrinsicsCamera2["camera_matrix"],
        intrinsicsCamera2["dist_coeffs"]
    )

    # The projected points should now be in the image coordinates of Camera 2
    imgpoints_from_1_to_2 = imgpoints_from_1_to_2.squeeze()

    # Verify by drawing the points onto Camera 2's image
    common_image_with_reprojects = draw_points(cv2.imread(commonImageFromCamera2Path), imgpoints_from_1_to_2, radius=4, color=(255, 0, 0))
    common_image_with_reprojects = draw_points(common_image_with_reprojects, imgPointsCamera2, radius=2, color=(0,255,0))
    saveImage(common_image_with_reprojects, "reprojected_points_on_camera2.jpg")

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
    print(f"image is {image.shape}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (rows, columns), None)

    if ret == True:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        corners2 = corners2.squeeze()
        # image_with_corners = cv2.drawChessboardCorners(image, (rows, columns), corners2, ret)
        image_with_corners = draw_points(image, corners2, radius=3)
        saveImage(image_with_corners, f"image_with_corners_{time.time()}")
        # breakpoint()
    else:
        print(f"ERROR, cannot find corners in {imagePath}")
        return None
    # Find the rotation and translation vectors.
    ret, rvecs, tvecs = cv2.solvePnP(
        objp, corners2, intrinsics["camera_matrix"], intrinsics["dist_coeffs"]
    )
    # breakpoint()
    print(f"TVECS {tvecs}")
    axis = np.float32([[300,0,0], [0,300,0], [0,0,-300]]).reshape(-1,3)
    imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, intrinsics["camera_matrix"], intrinsics["dist_coeffs"])
    new_image = draw(image, corners2, imgpts)
    print(f"new image {new_image.shape}")
    cv2.imshow("new image",new_image)
    # cv2.waitKey(0)
    return ret, rvecs, tvecs, objp, corners2 

def draw_points(image, landmarks2D, radius=2, color=(0, 0, 255)):
    for point in landmarks2D:
        # print(f"draw_points of point {point}")
        x, y = int(round(point[0])), int(round(point[1]))
        cv2.circle(image, (x, y), radius=radius, color=color, thickness=-1)
    return image

def saveImage(image, name):
    cv2.imshow("img", image)
    relative_path = os.path.join(
        script_dir, f"{name}.png"
    )
    cv2.imwrite(relative_path, image)
    cv2.waitKey(500)

def draw(img, corners, imgpts):
    # breakpoint()
    corner = (int(corners[0][0]), int(corners[0][1]))
    # tuple(corners[0].ravel())
    imgpts = imgpts.squeeze()
    img = cv2.line(img, corner, (int(imgpts[0][0]), int(imgpts[0][1])), (255,0,0), 5)
    img = cv2.line(img, corner,(int(imgpts[1][0]), int(imgpts[1][1])), (0,255,0), 5)
    img = cv2.line(img, corner, (int(imgpts[2][0]), int(imgpts[2][1])), (0,0,255), 5)
    return img

if __name__ == "__main__":
    main()