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
     
    ret, rvec, tvec, objPoints, imgPoints  = getExtrinsics(calibrator_image,lucid_intrinsics)
    # print(f"object points {objPoints}")
    # print(f"imgPoints {imgPoints}")
    
  
    # Calculate marker position in lucid system from image coordinates
 



    
    # Calculate marker position in MOCAP system from markers.csv as captures
    markers_csv = "./accuracyLucid/data/markers.csv"
    df = pd.read_csv(markers_csv)
    marker_1 = df.loc[0, ['x', 'y', 'z']]
    marker_1 = np.array(marker_1)
    marker_1_hom = np.append(marker_1, 1)
    # print(f"marker_1_hom is {marker_1_hom}")
    marker1_image_coordinates_hom = np.array([30,450,-12, 1])

    marker_1_in_lucid = get_marker_position_in_lucid(marker1_image_coordinates_hom, rvec, tvec)
    marker_1_in_mocap = get_marker_position_in_mocap(marker_1_hom)
    
    marker_2 = df.loc[1, ['x', 'y', 'z']]
    marker_2 = np.array(marker_2)
    marker_2_hom = np.append(marker_2, 1)
    # print(f"marker_2_hom is {marker_2_hom}")
    marker2_image_coordinates_hom =  np.array([630,90,-12, 1])
    marker_2_in_lucid = get_marker_position_in_lucid(marker2_image_coordinates_hom, rvec, tvec)
    marker_2_in_mocap = get_marker_position_in_mocap(marker_2_hom)



    
def get_marker_position_in_lucid(marker_image_coordinates_hom, rvec, tvec):
    transform_chessboard_to_lucid = np.eye(4)
    #TODO: check the T on rotation in main script
    transform_chessboard_to_lucid[:3, :3] = cv2.Rodrigues(rvec)[0]
    transform_chessboard_to_lucid[:3, 3] = tvec.T
    # print(f"transform_chessboard_to_lucid {transform_chessboard_to_lucid}")

    marker_in_lucid_system = transform_chessboard_to_lucid @ marker_image_coordinates_hom
    marker_in_lucid_system_hom = marker_in_lucid_system / marker_in_lucid_system[3]
    marker_in_lucid_system_hom /= 1000
    print(f"----------")
    print(f"marker_in_lucid_system_hom is {marker_in_lucid_system_hom}")
    print(f"----------")
    return marker_in_lucid_system_hom

def get_marker_position_in_mocap(marker_hom):    
    transform_file_mocap = json.load(open("./accuracyLucid/data/MotionCapture_CameraTransform.json"))
    transform_into_mocap_system = np.reshape(transform_file_mocap["cameraTransform"],(4,4)).T
    # print(f"transform_mocap_to_point_of_view is {transform_into_mocap_system}")
    
    marker_in_mocap_system = transform_into_mocap_system @ marker_hom
    marker_in_mocap_system_hom = marker_in_mocap_system / marker_in_mocap_system[3]
    
    print(f"----------")
    print(f"marker_in_mocap_system is {marker_in_mocap_system_hom}")
    print(f"----------")
    return marker_in_mocap_system_hom
    
    
if __name__ == "__main__":
    main()