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

from asyncio.windows_events import NULL
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

# import threading
# import queue
# import keyboard



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
landmarks3D = None
vertices = None
landmarks2Dfullres = None
rotation_vector = None
translation_vector = None
cameraFrame = None
script_dir = os.path.dirname(os.path.abspath(__file__))
face_detector = detectors.FAN()
r180x = np.array([[1,  0,  0],
                    [0, -1,  0],
                    [0,  0, -1]])
# A shared variable or queue to communicate between threads
# key_queue = queue.Queue()

def main(args):
    global global_args, landmarks3D, vertices, landmarks2Dfullres, cameraFrame, rotation_vector, translation_vector
    global_args = args

    # Hook the 'q' and 's' keys
    # keyboard.on_press_key('q', on_press_q)
    # keyboard.on_press_key('p', on_press_p)

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
        visualize()
        end_visualize_time = time.time()
        visualize_time = end_visualize_time - start_visualize_time

        print(
            f"Deca and SolvePNP time > {deca_and_solvepnp_time} [visualize:{visualize_time}] - dist {distance}"
        )
        
    # Don't forget to remove the hooks when you're done
    # keyboard.unhook_all()




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
    image_numpy = (image_tensor.permute(1, 2, 0).cpu().detach().numpy() * 255).astype(
        np.uint8
    )
    for point in landmarks2D:
        x, y = int(round(point[0])), int(round(point[1]))
        cv2.circle(image_numpy, (x, y), radius=5, color=(0, 0, 255), thickness=-1)

    bbox = imageData["bbox"]
    left, top, right, bottom = bbox
    cv2.rectangle(
        image_numpy, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2
    )


def visualize():
    global cameraFrame
    
    rotation_matrix = cv2.Rodrigues(rotation_vector)
    vertices_rotated = np.dot(vertices, rotation_matrix[0].T)
    vertices_rotated = np.dot(vertices_rotated, r180x)

    # # Translate original mesh in place where 3D landamrks are
    vertices_rotated_translated = vertices_rotated + translation_vector.T
    head_mesh.vertices = o3d.utility.Vector3dVector(vertices_rotated_translated)
    # head_mesh.vertices = o3d.utility.Vector3dVector(vertices_rotated)
    # head_mesh.vertices = o3d.utility.Vector3dVector(vertices_rotated)
    # head_mesh.compute_vertex_normals()  # If normals are not set, call this
    visualizer.update_geometry(head_mesh)

    visualizer.poll_events()
    visualizer.update_renderer()

    cameraFrame = draw_points(cameraFrame, landmarks2Dfullres)
    cv2.imshow(camera_window_name, cameraFrame)


def save_obj_with_landmarks3d():
    relative_path = os.path.join(script_dir,"..", "TestSamples", "camera_feed", "fromCameraFeed_mesh_expression.ply")
    head_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d.io.write_triangle_mesh(
        relative_path,
        head_mesh,
    )
def save_img_with_landmarks2d():
    relative_path = os.path.join(script_dir,"..", "TestSamples", "camera_feed", "fromCameraFeed_img_with_landmarks2d.png")
    frame_with_landmarks = draw_points(cameraFrame, landmarks2Dfullres)
    cv2.imwrite(relative_path, frame_with_landmarks)   

# def draw_points(image, landmarks2D, imageData):
def draw_points(image, landmarks2D):
    # image_numpy = (image.permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8)
    for point in landmarks2D:
        x, y = int(round(point[0])), int(round(point[1]))
        cv2.circle(image, (x, y), radius=5, color=(0, 0, 255), thickness=-1)

    # bbox = imageData['bbox']
    # left, top, right, bottom = bbox
    # cv2.rectangle(image_numpy, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
    # breakpoint()

    return image

def scalePoints(points, fromPointIndex, toPointIndex, desiredSize, desiredScalingFactor=None):
    if(fromPointIndex is not None and toPointIndex is not None):
         eyeDistance = math.dist(points[fromPointIndex], points[toPointIndex])
         scalingFactor = desiredSize/eyeDistance # 3 cm between real life corresponing points (eyes inner corners)
         points *= scalingFactor
    elif (desiredScalingFactor is not None):
        points *= desiredScalingFactor
        scalingFactor = desiredScalingFactor
    return scalingFactor

def start_visualizer():
    global visualizer, global_args, head_mesh
    # visualizer = o3d.visualization.Visualizer()
    visualizer = o3d.visualization.VisualizerWithKeyCallback()
    visualizer.create_window("Open3D - 3D Visualizer", 1024, 768)

    opt = visualizer.get_render_option()
    opt.background_color = np.asarray([1, 1, 1])  # Dark grey background color

    # Configure the camera's intrinsic and extrinsic parameters
    # (assuming you have predefined these parameters)
    camera_params = o3d.camera.PinholeCameraParameters()
    fx = fy = 640 / (2 * np.tan(np.deg2rad(60 / 2)))
    cx = 640 / 2
    cy = 480 / 2
    camera_params.intrinsic.set_intrinsics(640, 480, fx, fy, cx, cy)
    # Set extrinsic parameters (the 4x4 extrinsic matrix)
    camera_params.extrinsic = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 100],
        [0, 0, 0, 1]
    ])
    # Apply the camera parameters to the visualizer
    view_control = visualizer.get_view_control()
    view_control.convert_from_pinhole_camera_parameters(camera_params)
    # vis.get_view_control().set_up(camera_parameters)
    # vis.setup_camera(camera_params.intrinsic, camera_params.extrinsic)

    # To set a default view point, you could also use the look_at method
    # eye = (0, 0, -1)  # Camera position
    # center = (0, 0, 0)  # Look at point
    # up = (0, 1, 0)  # Up vector
    # visualizer.look_at(eye, center, up)

    add_gizmo(visualizer)
    add_wireframes(visualizer)




    
    head_mesh = o3d.io.read_triangle_mesh(global_args.templateMeshPath)
    vertices = np.asarray(head_mesh.vertices)
    scalePoints(vertices, 3827, 3619, 3)
    head_mesh.vertices = o3d.utility.Vector3dVector(vertices)

    head_mesh.compute_vertex_normals()
    visualizer.add_geometry(head_mesh)

    visualizer.register_key_callback(ord('F'), focus_camera_on_head)
    visualizer.register_key_callback(ord('Q'), on_press_q)
    visualizer.register_key_callback(ord('P'), on_press_p)

    # visualizer.add_geometry(coordinate_frame)

    visualizer.reset_view_point(True)
    # Run the visualization
    # visualizer.run()

# Function to update the view so that the head is fully visible
def focus_camera_on_head(vis):
    print("FOCUS")
     # Assuming 'head' is your head mesh object and 'bbox' is its axis-aligned bounding box
    # (You would replace the following line with the actual head mesh)
    bbox = head_mesh.get_oriented_bounding_box()

    # Compute the center of the head mesh and the extent to adjust the view
    center = bbox.get_center()
    extent = np.asarray(bbox.extent)

    # Get the view control and associated parameters
    view_control = vis.get_view_control()
    cam_params = view_control.convert_to_pinhole_camera_parameters()

    # Distance is based on the extent of the bounding box
    # Adjust 'distance_factor' as needed to ensure the head is fully visible
    distance_factor = max(extent) * 2
    distance = [0, 0, distance_factor]

    # Update the camera parameters to focus on the head
    cam_params.extrinsic = np.linalg.inv(
        o3d.camera.PinholeCameraParameters.create_look_at(
            center + distance, center, [0, 1, 0]
        )
    )

    # Apply the new camera parameters
    view_control.convert_from_pinhole_camera_parameters(cam_params)

    return False

# Function to add a visual gizmo (coordinate frame) to the center of the scene
def add_gizmo(vis):
    # Create a coordinate frame (gizmo) at the center of the scene
    gizmo = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
    vis.add_geometry(gizmo)
    
# Function to add wireframe planes to the scene
def add_wireframes(vis):
    # Create wireframe planes to represent the scene
    # Here is an example of creating a ground plane wireframe
    # plane_meshXZ = o3d.geometry.TriangleMesh.create_box(width=500.0, height=0.01, depth=500.0)
    # plane_meshXZ.compute_vertex_normals()
    # # plane_meshXZ.translate(np.array([0, -0.05, 0]))
    # plane_meshXZ.translate(np.array([0, -50, 0]))
    # plane_meshXZ.paint_uniform_color([1, 0, 0])
    # vis.add_geometry(plane_meshXZ)

    # plane_meshXY = o3d.geometry.TriangleMesh.create_box(width=500.0, height=500.0, depth=0.01)
    # plane_meshXY.compute_vertex_normals()
    # # plane_meshXY.translate(np.array([ 0,0, -0.05]))
    # plane_meshXY.translate(np.array([ 0,0, -50]))
    # plane_meshXY.paint_uniform_color([0.5, 0.5, 0.5])
    # vis.add_geometry(plane_meshXY)
    
    # plane_meshYZ = o3d.geometry.TriangleMesh.create_box(width=0.01, height=500.0, depth=500.0)
    # plane_meshYZ.compute_vertex_normals()
    # # plane_meshYZ.translate(np.array([ -0.05,0, 0]))
    # plane_meshYZ.translate(np.array([ -50,0, 0]))
    # plane_meshYZ.paint_uniform_color([0.5, 0.5, 0.5])
    # vis.add_geometry(plane_meshYZ)
    
    return False

def start_webcam():
    global camera, camera_window_name
    # Capture video from the first camera device
    camera = cv2.VideoCapture(0)
    # Set the desired resolution
    resolution_width = 1920
    resolution_height = 1080
    # resolution_width = 480
    # resolution_height = 270
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

def on_press_q(e):
    print("KEY q")
    stop_visualizer()
    stop_webcam()
    # keyboard.unhook_all()  # Remove all keyboard hooks

def on_press_p(e):
    print("KEY p")
    save_obj_with_landmarks3d()
    save_img_with_landmarks2d()

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