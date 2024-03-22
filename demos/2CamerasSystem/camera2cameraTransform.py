# from skimage.transform.hough_transform import label_distant_points
from operator import matmul
import cv2
import numpy as np
import os
import time
import yaml
import argparse

# Global vars
script_dir = os.path.dirname(os.path.abspath(__file__))
common_images = []
intrinsicsCamera1 = None
intrinsicsCamera2 = None
global_args = None
object_points_from_camera1 = []
object_points_from_camera2 = []
def main(args):
    global global_args
    global object_points_from_camera1, object_points_from_camera2
    global_args = args
    load_config()
    # for common_image in common_images:
    #     print(f" we have {common_image[0].shape} and {common_image[1].shape}")
    for image_number, common_image in enumerate(common_images):
        image_from_camera_1 = common_image[0]
        image_from_camera_2 = common_image[1]
        print(f"----------IMAGE {image_number}------------")
        print(f"NEW COUPLE OF IMAGES {image_from_camera_1.shape} and {image_from_camera_2.shape}")
        retCamera1, rvecsCamera1, tvecsCamera1, objPointsCamera1, imgPointsCamera1 = getExtrinsics(
            image_from_camera_1,
            intrinsicsCamera1,
            1,
            image_number
        )
        rotationMatrixCamera1 = cv2.Rodrigues(rvecsCamera1)[0].T
        distFromCamera1 = np.linalg.norm(tvecsCamera1) / 10  # in cm
        print(
            f"WEBCAM: ret {retCamera1}, \n"
            f"rotationMatrix {rotationMatrixCamera1}, \n"
            f"tVecs {tvecsCamera1}, \n"
            f"DIST:{distFromCamera1:.01f}cm"
        )

        retCamera2, rvecsCamera2, tvecsCamera2, objPointsCamera2, imgPointsCamera2 = getExtrinsics(
            image_from_camera_2,
            intrinsicsCamera2,
            2,
            image_number
        )
        rotationMatrixCamera2 = cv2.Rodrigues(rvecsCamera2)[0].T
        distFromCamera2 = np.linalg.norm(tvecsCamera2) / 10  # in cm
        print(
            f"LUCID: ret {retCamera2}, \n"
            f"rotationMatrix {rotationMatrixCamera2}, \n"
            f"tVecs {tvecsCamera2}, \n"
            f"DIST:{distFromCamera2:.01f}cm"
        )
        
        # Step 1: Transform object points to Camera 1's coordinate system
        transformed_objPoints_camera1 = apply_transform_to_object_points(objPointsCamera1, rvecsCamera1, tvecsCamera1)
        print(f" points are {transformed_objPoints_camera1.shape}")
        object_points_from_camera1.extend(transformed_objPoints_camera1)
        transformed_objPoints_camera2 = apply_transform_to_object_points(objPointsCamera2, rvecsCamera2, tvecsCamera2)
        print(f" points are {transformed_objPoints_camera2.shape}")
        object_points_from_camera2.extend(transformed_objPoints_camera2)
        
        # Step 2: Compose transformations from Camera 1 to Camera 2
        # R_composed, t_composed = get_composed_matrix_from_1_to_2(rvecsCamera1, tvecsCamera1, rvecsCamera2, tvecsCamera2)
        
        # Transform object points To 
        # reverseProjection(
        #     objPointsCamera1, 
        #     R_composed, t_composed,
        #     intrinsicsCamera2, 
        #     image_from_camera_2,imgPointsCamera2,
        #     image_number)
    print(f"-------END--------")
    object_points_from_camera1 = np.asarray(object_points_from_camera1)
    object_points_from_camera2 = np.asarray(object_points_from_camera2)
    print(f"We have collected {object_points_from_camera1.shape} and {object_points_from_camera2.shape}")
    transformation_matrix, rotation, translation = find_transformation_matrix(object_points_from_camera1, object_points_from_camera2)
    print(f"transformation_matrix IS {transformation_matrix}")
    
    # double check by reverse projecting object points onto camera 2 image but using this transform matrix
    for image_number, common_image in enumerate(common_images):
        image_from_camera_1 = common_image[0]
        image_from_camera_2 = common_image[1]
        # calculate the slicing indexes
        slice_start_index = image_number * 88
        slice_end_index = slice_start_index + 88
        
        object_points_slice = object_points_from_camera1[slice_start_index:slice_end_index, :]
        reverseProjection(object_points_slice, rotation, translation, intrinsicsCamera2, image_from_camera_2,None,image_number )

def find_transformation_matrix(A, B):
    """
    Finds the transformation matrix that best approximates the transformation needed to align two sets of 3D points.
    
    Parameters:
    - A: numpy.ndarray, shape (n_points, 3)
        The source point set.
    - B: numpy.ndarray, shape (n_points, 3)
        The destination point set, where we want to align A.
    
    Returns:
    - T: numpy.ndarray, shape (4, 4)
        The transformation matrix that best aligns A to B.
    """
    # Compute the centroids of both sets
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    
    # Center the points around the origin
    A_centered = A - centroid_A
    B_centered = B - centroid_B
    
    # Compute the covariance matrix
    H = A_centered.T @ B_centered
    
    # Perform Singular Value Decomposition
    U, S, Vt = np.linalg.svd(H)
    
    # Compute the rotation matrix
    R_matrix = Vt.T @ U.T
    
    # Ensure a right-handed coordinate system
    if np.linalg.det(R_matrix) < 0:
       Vt[2, :] *= -1
       R_matrix = Vt.T @ U.T
    
    # Compute the translation vector
    translation = centroid_B - R_matrix @ centroid_A
    
    # Construct the transformation matrix
    T = np.identity(4)
    T[:3, :3] = R_matrix
    T[:3, 3] = translation
    
    return T, R_matrix, translation
def reverseProjection(
        objPoints_camera1, 
        R_composed, t_composed, 
        intrinsicsCamera2, 
        commonImageFromCamera2, imgPointsCamera2,
        image_number):
    
   
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
    common_image_with_reprojects = draw_points(
        commonImageFromCamera2, 
        imgpoints_from_1_to_2, 
        radius=4, 
        color=(255, 0, 0))
    if(imgPointsCamera2 is not None):
        common_image_with_reprojects = draw_points(
            common_image_with_reprojects, 
            imgPointsCamera2, 
            radius=2, 
            color=(0,255,0))
    saveImage(common_image_with_reprojects, f"image_{image_number}_reprojected_points_on_camera2")

def apply_transform_to_object_points(objp, rvecs, tvecs):
    # objPoints_camera1 = np.dot(R_camera1, objp.T) + tvecsCamera1
    r_matrix, _ = cv2.Rodrigues(rvecs)
    new_objp = np.dot(r_matrix, objp.T) + tvecs
    # breakpoint()
    return new_objp.T
    
def get_composed_matrix_from_1_to_2(rvecsCamera1, tvecsCamera1, rvecsCamera2, tvecsCamera2):

    R_camera1, _ = cv2.Rodrigues(rvecsCamera1)
    # Invert Camera 1's transformation
    R_camera1_inv = R_camera1.T
    t_camera1_inv = -np.dot(R_camera1_inv, tvecsCamera1)

    # Apply Camera 2's transformation
    R_camera2, _ = cv2.Rodrigues(rvecsCamera2)
    t_camera2 = tvecsCamera2

    # Now, compose the transformations to map the points to Camera 2's coordinate system
    R_composed = np.dot(R_camera2, R_camera1_inv)
    t_composed = np.dot(R_camera2, t_camera1_inv) + t_camera2
    return R_composed, t_composed

def getExtrinsics(image, intrinsics, camera_number, image_number):
    # rows = 6
    # columns = 9    
    rows = 11
    columns = 8
    # Termination criteria for the corner sub-pixel algorithm
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    # Define the size of chessboard squares in millimeters.
    # square_size = 23
    square_size = 60
    # Prepare object points with actual square size.
    objp = objp * square_size

    print(f"image is {image.shape}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (rows, columns), None)

    if ret == True:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        corners2 = corners2.squeeze()
        # image_with_corners = cv2.drawChessboardCorners(image, (rows, columns), corners2, ret)
        image_with_corners = draw_points(image, corners2, radius=3)
        saveImage(image_with_corners, f"image_{image_number}_with_corners_camera{camera_number}")
        # breakpoint()
    else:
        print(f"ERROR, cannot find corners in {image.shape}")
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
def load_config():
    global common_images, intrinsicsCamera1, intrinsicsCamera2
    with open(get_path(global_args.config), "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return
    for common_image in config["image_paths"]:
        image1_path = get_path(common_image["fromCamera1"])
        image2_path = get_path(common_image["fromCamera2"])
        image1 = cv2.imread(image1_path)
        image2 = cv2.imread(image2_path)
        if image1 is not None and image2 is not None:
            print(f"loading images: {image1.shape} - {image2.shape}")
            common_images.append((image1, image2))
        else:
            print(f"Failed to load images from {image1_path} or {image2_path}")

    intrinsics = config["intrinsics"]
    intrinsicsCamera1 = np.load(get_path(intrinsics["camera1"]))
    intrinsicsCamera2 = np.load(get_path(intrinsics["camera2"]))

def get_path(path):
    # Check if the path starts with a '.'
    if path.startswith('.'):
        path = path[2:]
        # It's a relative path, so join it with the script directory
        # print(f"path {path} - script_dir {script_dir} >>> {os.path.join(script_dir, path)}")
        return os.path.join(script_dir, path)
    else:
        # It's an absolute path or a relative path not starting with '.', return as-is
        return path
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="camera2cameraTransform")
    
    parser.add_argument(
        "--config",
        default="./config.yaml",
        type=str,
        help="path to the config file",
    )
    main(parser.parse_args())