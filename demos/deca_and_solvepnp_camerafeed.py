# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms
# in the LICENSE file included with this software distribution.
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import os, sys
from re import S
from turtle import st
import cv2
import numpy as np
from time import time
from scipy.io import savemat
import argparse
from tqdm import tqdm
import torch
import math

import time
import open3d as o3d

import trimesh
import open3d as o3d


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from decalib.deca import DECA

from decalib.datasets import datasets, detectors
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg
from decalib.utils.tensor_cropper import transform_points


# GLOBAL VARIABLES
visualizer = None
camera = None
global_args = None
head_mesh = None
camera_window_name = None
face_detector = detectors.FAN()


def main(args):
    global global_args
    global_args = args


    # Load camera matrix
    calibration_data = args.calibrationData
    data = np.load(calibration_data)
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]

    # Ensure that camera_matrix and dist_coeffs are also float32 and have the correct shapes
    camera_matrix = np.asarray(camera_matrix, dtype=np.float32)
    dist_coeffs = np.asarray(dist_coeffs, dtype=np.float32)

    print("camera_matrix", camera_matrix, "\n dist_coeffs", dist_coeffs)

    # DECA

    # whether to use FLAME texture model to generate uv texture map
    deca_cfg.model.use_tex = False

    # pytorch3d or standard(default)
    deca_cfg.rasterizer_type = "pytorch3d"
    # whether to extract texture from input image as the uv texture map, set false if you want albedo map from FLAME model
    deca_cfg.model.extract_tex = True
    
    device = "cuda"
    deca = DECA(config=deca_cfg, device=device)

    start_visualizer()
    start_webcam()
    
    deca_and_solvepnp_time = 0
    start_time = 0
    end_time = 0
    
    torch.set_grad_enabled(False)

    while True:
        # Read a frame
        ret, cameraFrame = camera.read()

        # If frame is read correctly, ret is True
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        start_time = time.time()
        # Convert frame to RGB (OpenCV uses BGR by default)
        # cameraFrame = cv2.cvtColor(cameraFrame, cv2.COLOR_BGR2RGB)

        imagedata = datasets.CameraData(cameraFrame, face_detector)[0]
        image = imagedata["image"].to(device)[None, ...]

        tform = imagedata["tform"][None, ...]
        tform = torch.inverse(tform).transpose(1, 2).to(device)
        # print("TFORM ", tform.shape, tform)
        # print("TFORM ", tform.shape)
        original_image = imagedata["original_image"][None, ...].to(device)

        _, image_height, image_width = imagedata["original_image"].shape

        # ---- ENCODING ----
        codedict = deca.encode(image)

        # ---- DECODING ----

        opdict = deca.decode_fast(
            codedict,
            original_image=original_image,
            tform=tform,
            use_detail=False,
            return_vis=False,
            full_res_landmarks2D=True,
        )

        # ---- LANDMARKS ----

        landmarks3D = opdict["landmarks3d_world"][0].cpu().numpy()[:, :3]
        scaling_factor = scalePoints(landmarks3D, 39, 42, 3)
        landmarks3D = np.ascontiguousarray(landmarks3D, dtype=np.float32)

        landmarks2Dfullres = opdict["landmarks2d_full_res"][0]
        landmarks2Dfullres = np.ascontiguousarray(landmarks2Dfullres, dtype=np.float32)

        vertices = opdict["verts"][0].cpu().numpy()[:, :3]
        scalePoints(vertices, None, None, None, scaling_factor)

        # ---- SOLVEPNP ----
        success, rotation_vector, translation_vector = cv2.solvePnP(
            landmarks3D,
            # landmarks2D,
            landmarks2Dfullres,
            camera_matrix,
            dist_coeffs,
        )
        if success:
            distance = np.linalg.norm(translation_vector)
        else:
            print(f"FAILED")
        # breakpoint()

        end_time = time.time()
        deca_and_solvepnp_time = end_time - start_time
        
        start_visualize_time = time.time()
        visualize(landmarks2Dfullres, landmarks3D, vertices, rotation_vector, translation_vector, cameraFrame)
        end_visualize_time = time.time()
        visualize_time = end_visualize_time - start_visualize_time
            
        print(f"Deca and SolvePNP time > {deca_and_solvepnp_time} [visualize:{visualize_time}]")

        if cv2.waitKey(1) == ord('q'):
            stop_visualizer()
            stop_webcam()
            break
        if cv2.waitKey(1) == ord('s'):
            save_obj_with_landmarks3d(landmarks3D, vertices)
            break
        


def scalePoints(
    points, fromPointIndex, toPointIndex, desiredSize, desiredScalingFactor=None
):
    if fromPointIndex is not None and toPointIndex is not None:
        eyeDistance = math.dist(points[fromPointIndex], points[toPointIndex])
        scalingFactor = (
            desiredSize / eyeDistance
        )  # 3 cm between real life corresponing points (eyes inner corners)
        points *= scalingFactor
    elif desiredScalingFactor is not None:
        points *= desiredScalingFactor
        scalingFactor = desiredScalingFactor
    return scalingFactor



def draw_points(i, image_tensor, landmarks2D, imageData, args):
    image_numpy = (image_tensor.permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8)
    for point in landmarks2D:
         x, y = int(round(point[0])), int(round(point[1])) 
         cv2.circle(image_numpy, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
         
    bbox = imageData['bbox']
    left, top, right, bottom = bbox
    cv2.rectangle(image_numpy, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)


def visualize(landmarks2Dfullres, landmarks3D, vertices, rotation_vector, translation_vector, frame):
    global visualizer, camera_window_name

    # rotation_matrix = cv2.Rodrigues(rotation_vector)
    # print("Rotation is ", rotation_matrix[0])
    # landmarks_3D_rotated = np.dot(landmarks3D, rotation_matrix[0].T)
    # vertices_rotated = np.dot(vertices, rotation_matrix[0].T)
    # # original_mesh.vertices = np.dot(original_mesh.vertices, rotation_matrix[0].T)
    # # breakpoint()
    head_mesh.vertices = o3d.utility.Vector3dVector(vertices)

    # # Translate original mesh in place where 3D landamrks are
    # # original_mesh.vertices += translation_vector.T
    # head_mesh.vertices = o3d.utility.Vector3dVector(vertices_rotated)
    head_mesh.compute_vertex_normals()  # If normals are not set, call this
    visualizer.update_geometry(head_mesh)
    visualizer.poll_events()
    visualizer.update_renderer()
    
    frame = draw_points(frame, landmarks2Dfullres)
    cv2.imshow(camera_window_name, frame)
    
def save_obj_with_landmarks3d(landmarks3D, vertices):
    
    head_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d.io.write_triangle_mesh(r'C:\Users\massimo.bortolamei\Documents\DECA\TestSamples\camera_feed\fromCameraFeed_mesh_expression.ply', head_mesh)

# def draw_points(image, landmarks2D, imageData):
def draw_points(image, landmarks2D):
    # image_numpy = (image.permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8)
    for point in landmarks2D:
         x, y = int(round(point[0])), int(round(point[1])) 
         cv2.circle(image, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
         
    # bbox = imageData['bbox']
    # left, top, right, bottom = bbox
    # cv2.rectangle(image_numpy, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
    #breakpoint()

    return image



def start_visualizer():
    global visualizer, global_args, head_mesh
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()

    opt = visualizer.get_render_option()
    opt.background_color = np.asarray([0.1, 0.1, 0.1])  # Dark grey background color

    head_mesh = o3d.io.read_triangle_mesh(global_args.templateMeshPath)
    head_mesh.compute_vertex_normals()
    visualizer.add_geometry(head_mesh)
    # Run the visualization
    # visualizer.run()



def start_webcam():
    global camera, camera_window_name
    # Capture video from the first camera device
    camera = cv2.VideoCapture(0)
    # Set the desired resolution
    # resolution_width = 1920
    # resolution_height = 1080
    resolution_width = 480
    resolution_height = 270
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolution_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution_height)
    # Check if the video stream is opened successfully
    if not camera.isOpened():
        print("Error: Could not open video stream.")
        exit()

    # Set the window name
    camera_window_name = "Webcam Augmented Feed"

    # Create a window to display the frames
    cv2.namedWindow(camera_window_name, cv2.WINDOW_NORMAL)
    # Set the desired window size
    desired_width = 400
    desired_height = 400
    cv2.resizeWindow(camera_window_name, desired_width, desired_height)


def stop_visualizer():
    global visualizer
    visualizer.destroy_window()


def stop_webcam():
    global camera
    # Release the video capture object and close all windows
    camera.release()
    # cv2.destroyAllWindows()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DECA + SolvePnP")


    # Additional Arguments not in original lib
    parser.add_argument(
        "--templateMeshPath",
        default="data\head_template_centered.ply",
        type=str,
        help="path to the ply template mesh for visualisation",
    )
    parser.add_argument(
        "--calibrationData",
        default="Python Camera Calibration/calibration_data_webcam_side.npz",
        type=str,
        help="calibration data with camera matrix and distortion coefficients as result of manual calibration",
    )

    main(parser.parse_args())