import numpy as np
import sys
import cv2
import pandas as pd
import json
# sys.path.insert(0,"../CamerasSystem2")
# sys.path.insert(0,"D:\asyvir\repositories\DECA\demos\CamerasSystem2")

from CamerasSystem2.camera2cameraTransform import getExtrinsics

def main():
    np.set_printoptions(suppress=True)
    calibrator_image = cv2.imread("./accuracyLucid/data/calibrator_from_lucid_marker_outside.jpg")
    lucid_intrinsics = np.load("./accuracyLucid/data/calibration_data_lucid_finalOptic_fullres.npz")
     
    ret, rvec, tvecs, objPoints, imgPoints  = getExtrinsics(calibrator_image,lucid_intrinsics)
    print(f"object points {objPoints}")
    print(f"imgPoints {imgPoints}")
    # point where the marker is, in objPoints: 630,90,-20
    marker_image_coordinates_hom = np.array([630,90,-10, 1])
    # marker_image_coordinates_hom = np.array([0,0,0, 1])
    transform_chessboard_to_lucid = np.eye(4)

    #TODO: check the T on rotation in main script
    transform_chessboard_to_lucid[:3, :3] = cv2.Rodrigues(rvec)[0]
    transform_chessboard_to_lucid[:3, 3] = tvecs.T
    print(f"transform_chessboard_to_lucid {transform_chessboard_to_lucid}")
    
    # marker_in_lucid_system = marker_image_coordinates_hom @ transform_chessboard_to_lucid
    marker_in_lucid_system = transform_chessboard_to_lucid @ marker_image_coordinates_hom
    marker_in_lucid_system_hom = marker_in_lucid_system / marker_in_lucid_system[3]
    marker_in_lucid_system_hom /= 1000
    print(f"----------")
    print(f"marker_in_lucid_system is {marker_in_lucid_system_hom}")
    print(f"----------")
    
    markers_csv = "./accuracyLucid/data/markers.csv"
    df = pd.read_csv(markers_csv)
    mean_vector = df.head(5)[['x', 'y', 'z']].mean()
    mean_vector_np = np.array(mean_vector)
    mean_vector_np_hom = np.append(mean_vector_np, 1)
    print(f"mean_vector_np is {mean_vector_np_hom}")
    

    transform_file_mocap = json.load(open("./accuracyLucid/data/Flex3_CameraTransform.json"))
    transform_into_mocap_system = np.reshape(transform_file_mocap["pointofview_5"],(4,4)).T
    print(f"transform_mocap_to_point_of_view is {transform_into_mocap_system}")
    
    # marker_in_mocap_system = mean_vector_np_hom @ transform_into_mocap_system
    marker_in_mocap_system = transform_into_mocap_system @ mean_vector_np_hom
    marker_in_mocap_system_hom = marker_in_mocap_system / marker_in_mocap_system[3]
    
    print(f"----------")
    print(f"marker_in_mocap_system is {marker_in_mocap_system_hom}")
    print(f"----------")
    
    transform_mocap_to_lucid = np.reshape(transform_file_mocap["cameraTransform"],(4,4)).T

    
    
    
if __name__ == "__main__":
    main()