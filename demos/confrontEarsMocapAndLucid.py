import numpy as np
import sys
import cv2
import pandas as pd
import json

from checkAccuracy import get_marker_position_in_mocap

def main():
    np.set_printoptions(suppress=True)

    print(f"----------")
   
    marker_1 = np.array([0.225468, 1.674136,-0.312837])
    marker_1_hom = np.append(marker_1, 1)
    marker_1_in_mocap = get_marker_position_in_mocap(marker_1_hom)
    marker_1_in_DECA = np.array([3.586317658543086,2.1549211733398783,221.5349203768601])
    marker_1_in_DECA/= 100

    marker_2 = np.array([0.218791, 1.676175, -0.486235])
    marker_2_hom = np.append(marker_2, 1)
    marker2_image_coordinates_hom =  np.array([630,90,-12, 1])
    marker_2_in_mocap = get_marker_position_in_mocap(marker_2_hom)
    marker_2_in_DECA = np.array([-7.650745068094491,1.4793079255525154,221.44403288569276])
    marker_2_in_DECA/= 100
    print(f"----------")

    print(f"Marker 1 from DECA is {marker_1_in_DECA}")
    print(f"Marker 2 from DECA is {marker_2_in_DECA}")
    print(f"----------")

    distanceBetweenMarkersDECA = np.linalg.norm(marker_2_in_DECA - marker_1_in_DECA)

    distanceBetweenMarkersMocap = np.linalg.norm(marker_2_in_mocap - marker_1_in_mocap)
    print(f"distanceBetweenMarkersDECA is {distanceBetweenMarkersDECA}")
    print(f"distanceBetweenMarkersMocap is {distanceBetweenMarkersMocap}")
    print(f"----------")




    
if __name__ == "__main__":
    main()