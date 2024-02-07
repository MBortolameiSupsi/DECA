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
import cv2
import numpy as np
from time import time
import argparse
import torch
import math

import time
import open3d as o3d

import open3d as o3d

from datetime import datetime
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from decalib.deca import DECA

from decalib.datasets import datasets, detectors
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg
from decalib.utils.tensor_cropper import transform_points


# GLOBAL VARIABLES
camera_matrix = None
camera_dist_coeffs = None
camera_height = None
camera_width = None
camera_index = None
image_folder = None
source = None
output_folder = None

visualizer2d = None
visualizer2d_width = None
visualizer2d_height = None

mirror_results = None
apply_to_head_mesh = None

save_vertices_rotated_and_translated = None
save_solvepnp_rototranslation = None
save_mesh_expression_with_landmarks3d = None
save_image_with_landmarks2d = None

global_args = None
head_mesh = None
input_image = None
deca_and_solvepnp_time = None

visualizer3d = None
camera = None
camera_window_name = None

landmarks3D = None
vertices = None
landmarks2Dfullres = None
rotation_vector = None
translation_vector = None
transform_matrix = np.eye(4)
distance = None
vertices_rotated_translated = None
desired_eye_distance = None
tmirror = np.array([[-1], [-1], [-1]])
face_detector = detectors.FAN()

script_dir = os.path.dirname(os.path.abspath(__file__))

deca = None
device = "cuda"


def main(args):
    global global_args, deca
    global input_image, transform_matrix, distance
    global deca_and_solvepnp_time
    global_args = args
    load_config()
    # DECA setup

    # whether to use FLAME texture model to generate uv texture map
    deca_cfg.model.use_tex = False
    # pytorch3d or standard(default)
    deca_cfg.rasterizer_type = "pytorch3d"
    # whether to extract texture from input image as the uv texture map, set false if you want albedo map from FLAME model
    deca_cfg.model.extract_tex = True

    deca = DECA(config=deca_cfg, device=device)

    load_head()

    if visualizer3d:
        start_visualizer3d()
    if visualizer2d:
        start_visualizer2d()

    if source["type"] == "camera":
        start_webcam()
    if source["type"] == "folder":
        images_from_folder = get_images_from_folder()

    deca_and_solvepnp_time = 0
    start_time = 0
    end_time = 0

    torch.set_grad_enabled(False)

    while True:
        if source["type"] == "camera":
            ret, input_image = camera.read()

            # If frame is read correctly, ret is True
            if not ret:
                print("Error: Can't receive frame (stream end?). Exiting ...")
                break
        elif source["type"] == "folder":
            if images_from_folder.__len__() > 0:
                input_image = images_from_folder.pop(0)
                print(
                    f">>>>>> input_image is {input_image.shape}, remaining {images_from_folder.__len__()}"
                )
            else:
                print("All images processed (or No Input Image at all)")
                return

        start_time = time.time()

        # ----------------------------
        success = deca_and_solvepnp(input_image)
        # ----------------------------

        end_time = time.time()
        deca_and_solvepnp_time = end_time - start_time

        if success:
            start_visualize_time = time.time()
            distance = np.linalg.norm(translation_vector)
            transform_matrix = compose_transform_matrix()
            
            if visualizer3d:
                visualize3d()
            if visualizer2d:
                visualize2d()
            end_visualize_time = time.time()
            visualize_time = end_visualize_time - start_visualize_time


            if save_mesh_expression_with_landmarks3d and source["type"] == "folder":
                save_img_2d()
            if save_image_with_landmarks2d and source["type"] == "folder":
                save_mesh_3d()
            if save_solvepnp_rototranslation:
                save_solvepnp_transform()
            print(
                f"Deca and SolvePNP time > {deca_and_solvepnp_time} [visualize:{visualize_time}] - dist {distance}"
            )
            print(f"transform matrix {transform_matrix}")
    
    # Don't forget to remove the hooks when you're done
    # keyboard.unhook_all()


def deca_and_solvepnp(input_image):
    global landmarks3D, vertices, landmarks2Dfullres
    global translation_vector, rotation_vector

    imagedata = datasets.CameraData(input_image, face_detector)[0]
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
    scaling_factor = scalePoints(landmarks3D, 39, 42, desired_eye_distance)
    landmarks3D = np.ascontiguousarray(landmarks3D, dtype=np.float32)

    landmarks2Dfullres = opdict["landmarks2d_full_res"][0]
    landmarks2Dfullres = np.ascontiguousarray(landmarks2Dfullres, dtype=np.float32)

    vertices = opdict["verts"][0].cpu().numpy()[:, :3]
    scalePoints(vertices, desiredScalingFactor=scaling_factor)

    # ---- SOLVEPNP ----
    success, rotation_vector, translation_vector = cv2.solvePnP(
        landmarks3D,
        # landmarks2D,
        landmarks2Dfullres,
        camera_matrix,
        camera_dist_coeffs,
    )

    return success
    # breakpoint()

def visualize3d():
    global head_mesh
    rotation_matrix = cv2.Rodrigues(rotation_vector)[0].T
    rotation_matrix[0, :] *= 1  # Invert X-axis
    rotation_matrix[1, :] *= -1  # Invert Y-axis
    rotation_matrix[2, :] *= -1  # Invert Z-axis

    vertices_rotated = np.matmul(vertices, rotation_matrix)
    mirrored_translation = translation_vector * tmirror
    vertices_rotated_translated = vertices_rotated + mirrored_translation.T

    head_mesh.vertices = o3d.utility.Vector3dVector(vertices_rotated_translated)

    visualizer3d.update_geometry(head_mesh)

    visualizer3d.poll_events()
    visualizer3d.update_renderer()


def visualize2d():
    global input_image
    input_image = draw_points(input_image, landmarks2Dfullres)
    text = f"Dist. cm: {distance:.3f} \n Time s:{deca_and_solvepnp_time:.3f}"
    draw_text(input_image, text)
    cv2.imshow(camera_window_name, input_image)

def draw_text(input_image, text):
    # Define the font for the text
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Define the starting position for the text (top right corner)
    base_x = input_image.shape[1] - 280
    base_y = 40  # You may need to adjust these values
    # Define the font scale and color
    font_scale = 1
    font_color = (0, 255, 0)  # Green color
    line_type = 2
    line_spacing = 30  # Adjust the line spacing based on font size

    # Split the text by '\n' to handle multiple lines
    lines = text.split('\n')
    
    # Draw each line on the image
    for i, line in enumerate(lines):
        line_position = (base_x, base_y + i * line_spacing)
        cv2.putText(input_image, 
                    line,  # Text string for the current line
                    line_position, 
                    font, 
                    font_scale, 
                    font_color, 
                    line_type)

def save_mesh_3d():
    global head_mesh
    now = datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")
    relative_path = os.path.join(
        script_dir,
        output_folder,
        f"mesh_expression_{date_time}.ply",
    )
    # head_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d.io.write_triangle_mesh(
        relative_path,
        head_mesh,
    )


def save_img_2d():
    now = datetime.now()
    # This will give you time precise up to milliseconds
    date_time = now.strftime("%Y%m%d_%H%M%S%f")[:17]  
    relative_path = os.path.join(
        script_dir,
        output_folder,
        f"image_landmarks2d_{date_time}.png",
    )
    print(f"saving {date_time}")
    frame_with_landmarks = draw_points(input_image, landmarks2Dfullres)
    cv2.imwrite(relative_path, frame_with_landmarks)

def compose_transform_matrix():
    global transform_matrix
    transform_matrix[:3, :3] = cv2.Rodrigues(rotation_vector)[0].T
    transform_matrix[:3, 3] = translation_vector.T
    return transform_matrix

def save_solvepnp_transform():
    with open(output_folder+f"/solvepnp_transform.txt", 'ab') as file:
        np.savetxt(file, transform_matrix)

def draw_points(image, landmarks2D):
    for point in landmarks2D:
        x, y = int(round(point[0])), int(round(point[1]))
        cv2.circle(image, (x, y), radius=2, color=(0, 0, 255), thickness=-1)
    return image


def getDesiredEyeDistance(vertices, fromPointIndex, toPointIndex):
    # this will be the desired size to which scale the landmarks
    return math.dist(vertices[fromPointIndex], vertices[toPointIndex])


def scalePoints(
    points,
    fromPointIndex=None,
    toPointIndex=None,
    desiredDistance=None,
    desiredScalingFactor=None,
):
    if fromPointIndex != None and toPointIndex != None:
        currentEyeDistance = math.dist(points[fromPointIndex], points[toPointIndex])
        scalingFactor = desiredDistance / currentEyeDistance
        # print(f"SCALING - initial distance was {currentEyeDistance} - desired is {desiredDistance} : scalingFactor {scalingFactor}")
        points *= scalingFactor
    elif desiredScalingFactor != None:
        points *= desiredScalingFactor
        scalingFactor = desiredScalingFactor
    return scalingFactor


def add_gizmo(vis, origin=[0, 0, 0], size=1):
    # Create a coordinate frame (gizmo) at the center of the scene
    gizmo = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=origin)
    vis.add_geometry(gizmo)


def add_wireframes(vis):
    commonWidth = 100
    commonHeight = 100
    commonDepth = 100
    shortSide = 0.1
    plane_origin = np.array([-commonWidth / 2, -commonHeight / 2, -commonDepth])

    add_gizmo(visualizer3d, [0, 0, 0], size=5)
    # add_gizmo(visualizer, plane_origin, size=50)

    plane_meshXZ = o3d.geometry.TriangleMesh.create_box(
        width=commonWidth, height=shortSide, depth=commonDepth
    )
    plane_meshXZ.compute_vertex_normals()
    # plane_meshXZ.translate(np.array([0, -0.05, 0]))
    plane_meshXZ.translate(plane_origin)
    plane_meshXZ.paint_uniform_color([0.5, 0, 0])
    # vis.add_geometry(plane_meshXZ, reset_bounding_box=False)
    vis.add_geometry(plane_meshXZ)

    plane_meshXY = o3d.geometry.TriangleMesh.create_box(
        width=commonWidth, height=commonHeight, depth=shortSide
    )
    plane_meshXY.compute_vertex_normals()
    # plane_meshXY.translate(np.array([ 0,0, -0.05]))
    plane_meshXY.translate(plane_origin)
    plane_meshXY.paint_uniform_color([0, 0.5, 0])
    # vis.add_geometry(plane_meshXY, reset_bounding_box=False)
    vis.add_geometry(plane_meshXY)

    plane_meshYZ = o3d.geometry.TriangleMesh.create_box(
        width=shortSide, height=commonHeight, depth=commonDepth
    )
    plane_meshYZ.compute_vertex_normals()
    # plane_meshYZ.translate(np.array([ -0.05,0, 0]))
    plane_meshYZ.translate(plane_origin)
    plane_meshYZ.paint_uniform_color([0, 0, 0.5])
    # vis.add_geometry(plane_meshYZ, reset_bounding_box=False)
    vis.add_geometry(plane_meshYZ)

    return False


def start_visualizer3d():
    global visualizer3d
    visualizer3d = o3d.visualization.VisualizerWithKeyCallback()
    visualizer3d.create_window("Open3D - 3D Visualizer", 1024, 768)

    opt = visualizer3d.get_render_option()
    opt.background_color = np.asarray([0.8, 0.8, 0.8])

    add_wireframes(visualizer3d)

    eye = np.array([0, 0, 0])  # Camera position (x, y, z)
    center = np.array([0, 0, -100])  # Look at point
    up = np.array([0, 1, 0])  # Up vector

    view_control = visualizer3d.get_view_control()
    view_control.set_constant_z_far(10000)
    view_control.set_constant_z_near(0.1)
    view_control.set_lookat(center)
    view_control.set_front(
        eye - center
    )  # The front vector is the opposite of the view direction
    view_control.set_up(up)
    view_control.set_zoom(0.5)
    # view_control.translate(5,10,2)
    # view_control.change_field_of_view(0.20)

    head_mesh.compute_vertex_normals()
    visualizer3d.add_geometry(head_mesh, reset_bounding_box=False)

    visualizer3d.register_key_callback(ord("Q"), on_press_q)
    visualizer3d.register_key_callback(ord("P"), on_press_p)


def load_head():
    global head_mesh, desired_eye_distance
    head_mesh = o3d.io.read_triangle_mesh(head_mesh_path)
    vertices = np.asarray(head_mesh.vertices)
    # head mesh is already perfect scale. we save eye distance
    # to scale landmarks later on
    desired_eye_distance = getDesiredEyeDistance(vertices, 3827, 3619)
    print(f"Desired Eye distance is {desired_eye_distance}")


def start_webcam():
    global camera

    camera = cv2.VideoCapture(0)

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)

    if not camera.isOpened():
        print("Error: Could not open video stream.")
        exit()


def get_images_from_folder():
    valid_extensions = (".jpg", ".jpeg", ".png")
    all_files = os.listdir(image_folder)
    image_names = [
        file for file in all_files if file.lower().endswith(valid_extensions)
    ]
    images = [
        cv2.imread(os.path.join(image_folder, image_name)) for image_name in image_names
    ]
    print(f">>>>> got images {images.__len__()} {images}")
    return images


def start_visualizer2d():
    global camera_window_name
    # feed enabled and is camera

    camera_window_name = "Webcam Augmented Feed"

    cv2.namedWindow(camera_window_name, cv2.WINDOW_NORMAL)
    # Set the desired window size
    desired_width = visualizer2d_width
    desired_height = visualizer2d_height
    cv2.resizeWindow(camera_window_name, desired_width, desired_height)


def stop_visualizer3d():
    visualizer3d.destroy_window()


def stop_webcam():
    camera.release()


def on_press_q(e):
    print("KEY q")
    stop_visualizer3d()
    stop_webcam()


def on_press_p(e):
    print("KEY p")
    save_mesh_3d()
    save_img_2d()


def load_config():
    global camera_matrix, camera_dist_coeffs
    global camera_width, camera_height
    global source, camera_index, image_folder
    global output_folder, head_mesh_path
    global visualizer2d, visualizer3d
    global visualizer2d_height, visualizer2d_width
    global mirror_results, apply_to_head_mesh
    global save_mesh_expression_with_landmarks3d
    global save_image_with_landmarks2d
    global save_solvepnp_rototranslation
    
    with open(global_args.config, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return

    camera = config["camera"]
    camera_width = camera["capture_resolution_width"]
    camera_height = camera["capture_resolution_height"]
    camera_calibration_data = camera["calibration_data"]
    if camera_calibration_data != None or camera_calibration_data != "":
        camera_data = np.load(camera_calibration_data)
        if camera_data != None:
            camera_matrix = np.asarray(camera_data["camera_matrix"], dtype=np.float32)
            camera_dist_coeffs = np.asarray(
                camera_data["dist_coeffs"], dtype=np.float32
            )
        else:
            print("ERROR fetching camera data")
            return
    else:
        print("ERROR no camera data specified")
        return
    print(
        "\n LOADED camera_matrix",
        camera_matrix,
        "\n dist_coeffs",
        camera_dist_coeffs,
        f"\n w:{camera_width} h:{camera_height}",
    )

    source = config["source"]
    if source["type"] == "camera":
        camera_index = source["camera_index"]
    elif source["type"] == "folder":
        image_folder = source["folder"]

    output = config["output"]    
    output_folder = os.path.join(script_dir, output["folder"])
    # Check if the directory exists, if not, create it
    os.makedirs(output_folder, exist_ok=True)
    
    save_mesh_expression_with_landmarks3d = output["save_mesh_expression_with_landmarks3d"]
    save_image_with_landmarks2d = output["save_image_with_landmarks2d"]
    save_solvepnp_rototranslation = output["save_solvepnp_rototranslation"]
    
    head_mesh_path = config["head_mesh"]

    visualize = config["visualize"]
    visualizer2d = visualize["visualizer2d"]
    visualizer2d_width = visualize["visualizer2d_width"]
    visualizer2d_height = visualize["visualizer2d_height"]
    visualizer3d = visualize["visualizer3d"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DECA + SolvePnP")

    # Additional Arguments not in original lib
    parser.add_argument(
        "--config",
        default="./config.yaml",
        type=str,
        help="path to the config file",
    )

    main(parser.parse_args())