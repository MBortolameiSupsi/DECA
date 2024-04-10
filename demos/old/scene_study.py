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

script_dir = os.path.dirname(os.path.abspath(__file__))
templateMeshPath = os.path.join(script_dir, "..", "data", "head_template_centered.ply")
head_mesh = None
visualizer = None


def main():
    start_visualizer()
    while True:
        visualize()


def visualize():
    visualizer.poll_events()
    visualizer.update_renderer()


def start_visualizer():
    global visualizer, head_mesh
    # visualizer = o3d.visualization.Visualizer()
    visualizer = o3d.visualization.VisualizerWithKeyCallback()
    visualizer.create_window("Open3D - 3D Visualizer", 1024, 768)

    opt = visualizer.get_render_option()
    opt.background_color = np.asarray([1, 1, 1])  # Dark grey background color


   
    
   

    add_gizmo(visualizer)
    add_wireframes(visualizer)
    
    # view_control = visualizer.get_view_control()
    # camera_params = view_control.convert_to_pinhole_camera_parameters()
    # Configure the camera's intrinsic and extrinsic parameters
    # (assuming you have predefined these parameters)
    # width = 1920    
    # height = 1080
    # fx = fy = width / (2 * np.tan(np.deg2rad(60 / 2)))
    # cx = width / 2
    # cy = height / 2
    # intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    # # intrinsic.intrinsic_matrix = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    # # camera_params = o3d.camera.PinholeCameraParameters()
    # camera_params.intrinsic = intrinsic
    # camera_params.extrinsic = np.array(
    #     [
    #         [1, 0, 0, 5],
    #         [0, 1, 0, 5],
    #         [0, 0, 1, 5],
    #         [0, 0, 0, 1],
    #     ]
    # )
    # # Apply the camera parameters to the visualizer
    # view_control.convert_from_pinhole_camera_parameters(camera_params)
    # visualizer.update_renderer()

    
    # To set a default view point, you could also use the look_at method
    # Define the camera view
    eye = np.array([5, 5, 10])  # Camera position (x, y, z)
    center = np.array([0, 0, 0])  # Look at point
    up = np.array([0, 1, 0])  # Up vector

    # Set the view control parameters
    view_control = visualizer.get_view_control()
    view_control.set_lookat(center)
    view_control.set_front(eye - center)  # The front vector is the opposite of the view direction
    view_control.set_up(up)
    view_control.set_zoom(0.5)  # Adjust this value for the desired zoom level
    # view_control.translate(5,10,2)
    # view_control.change_field_of_view(0.20)
    head_mesh = o3d.io.read_triangle_mesh(templateMeshPath)
    vertices = np.asarray(head_mesh.vertices)
    scalePoints(vertices, 3827, 3619, 3)
    head_mesh.vertices = o3d.utility.Vector3dVector(vertices)

    head_mesh.compute_vertex_normals()
    visualizer.add_geometry(head_mesh, reset_bounding_box=False)


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


if __name__ == "__main__":
    main()