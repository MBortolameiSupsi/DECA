import numpy as np
import sys
import cv2
# sys.path.insert(0,"../CamerasSystem2")
# sys.path.insert(0,"D:\asyvir\repositories\DECA\demos\CamerasSystem2")

from CamerasSystem2.camera2cameraTransform import getExtrinsics

def main():
    calibrator_image = cv2.imread("./accuracyLucid/data/calibrator_from_lucid_marker_outside.jpg")
    lucid_intrinsics = np.load("./accuracyLucid/data/calibration_data_lucid_finalOptic_fullres.npz")
    ret, rvec, tvecs, objPoints, imgPoints  = getExtrinsics(calibrator_image,lucid_intrinsics)

if __name__ == "__main__":
    main()