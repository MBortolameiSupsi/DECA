from skimage.transform.hough_transform import label_distant_points
import cv2
import numpy as np
import os
import time

script_dir = os.path.dirname(os.path.abspath(__file__))


def main():
    commonImageFromCamera1Path = default = os.path.join(
        script_dir, "./commonImageFromWebcam_640x360.jpg"
    )
    commonImageFromCamera2Path = default = os.path.join(
        script_dir, "./commonImageFromLucid_640x360.jpg"
    )

    intrinsicsCamera1Path = default = os.path.join(
        script_dir, "./calibration_data_webcam_side_640_2.npz"
    )
    intrinsicsCamera2Path = default = os.path.join(
        script_dir, "./calibration_data_lucid3_640.npz"
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
    )

    # apply rtvec from camera 1 to object points
    # so that they are in world coordinate wrt to camera1 in 0
    transform_matrix_camera1 = np.eye(4)
    
    transform_matrix_camera1[:3, 3] = np.squeeze(tvecsCamera1)
    transform_matrix_camera1[:3, :3] = cv2.Rodrigues(rvecsCamera1)[0] 
    objPointsCamera1_homogeneous = np.hstack((objPointsCamera1, np.ones((objPointsCamera1.shape[0], 1))))
    objPointsCamera1_world = objPointsCamera1_homogeneous @ transform_matrix_camera1.T
    objPointsCamera1_world = objPointsCamera1_world[:, :3] / objPointsCamera1_world[:, 3, np.newaxis]
    
    # transform_matrix_camera2 = np.eye(4)
    # transform_matrix_camera2[:3, 3] = np.squeeze(tvecsCamera2)
    # transform_matrix_camera2[:3, :3] = cv2.Rodrigues(rvecsCamera2)[0] 
    # objPointsCamera2_homogeneous = np.hstack((objPointsCamera2, np.ones((objPointsCamera2.shape[0], 1))))
    # objPointsCamera2_world = objPointsCamera2_homogeneous @ np.linalg.inv(transform_matrix_camera2)
    # objPointsCamera2_world = objPointsCamera2_world[:, :3] / objPointsCamera2_world[:, 3, np.newaxis]
    breakpoint()

    # objPointsCamera1_world_1 = objPointsCamera1_world_1.T
    # print(f"objPointsCamera1_world after transform MATRIX is {objPointsCamera2_world}")    
    print(f"objPointsCamera1_world after transform MATRIX is {objPointsCamera1_world}")    

    # now that we have the obj points in world coord, wrt camera 1
    # we use projectPoints with camera2 rtvecs, mtx, dist
    # to see where they fall in the common image from camera2
    imgpointsFrom1To2, _ = cv2.projectPoints(
        objPointsCamera1_world,
        rvecsCamera2,
        tvecsCamera2,
        intrinsicsCamera2["camera_matrix"],
        intrinsicsCamera2["dist_coeffs"],
    )    
    # imgpointsFrom1To2, _ = cv2.projectPoints(
    #     objPointsCamera2_world,
    #     rvecsCamera2,
    #     tvecsCamera2,
    #     intrinsicsCamera2["camera_matrix"],
    #     intrinsicsCamera2["dist_coeffs"],
    # )
    imgpointsFrom1To2 = np.array(imgpointsFrom1To2.squeeze())
    print(f"projected points are {imgpointsFrom1To2}")
    # common_image_with_reprojects = draw_points(cv2.imread(commonImageFromCamera2Path), imgpointsFrom1To2, radius=4, color=(255,0,0))
    common_image_with_reprojects = draw_points(cv2.imread(commonImageFromCamera1Path), imgpointsFrom1To2, radius=4, color=(255,0,0))
    # common_image_with_reprojects = draw_points(common_image_with_reprojects, imgPointsCamera2, radius=2, color=(0,255,0))
    saveImage(common_image_with_reprojects, f"reproject")


def getExtrinsics(imagePath, intrinsics):
    rows = 11
    columns = 8
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

if __name__ == "__main__":
    main()