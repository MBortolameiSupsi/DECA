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
import cv2
import numpy as np
from time import time
import argparse
import torch
import math
from skimage.transform import estimate_transform, warp, resize, rescale
import mediapipe as mp

import time
from datetime import datetime

import open3d as o3d
import yaml
import csv
from memory_profiler import profile
from memory_profiler import memory_usage

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from decalib.deca import DECA

from decalib.datasets import datasets, detectors
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg
from decalib.utils.tensor_cropper import transform_points

from arena_api.system import system
from arena_api.buffer import *
import ctypes


script_dir = os.path.dirname(os.path.abspath(__file__))
# GLOBAL VARIABLES from Configs
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
save_solvepnp_data = None
save_mesh_expression_with_landmarks3d = None
save_image_with_landmarks2d = None
save_ear_points = None
ear_csv = None
csv_writer = None
logs_txt = None
images_txt = None

save_video_feed = None

# GLOBAL VARIABLES 
global_args = None
should_exit = False
camera_window_name = None

deca_and_solvepnp_time = None
time_logs = None
fps = None
avg_fps = None

visualizer3d = None
camera = None
lucidCamera = None

tmirror = np.array([[-1], [-1], [-1]])


deca = None
device = "cuda"

mp_face_mesh = mp.solutions.face_mesh
# model = mp_face_mesh.FaceMesh()
model = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        # min_detection_confidence=0.7,
        # min_tracking_confidence=0.7   
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
        )
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=1,  # 0 for short-range model, suitable for most cases
    min_detection_confidence=0.5)


SAFE_COPY = True

def main(args):
    global global_args, deca
    global deca_and_solvepnp_time, fps
    global_args = args
    load_config()
    # DECA setup
    # pytorch3d or standard(default)
    deca_cfg.rasterizer_type = "pytorch3d"
    deca = DECA(config=deca_cfg, device=device)

    # head model, already scaled, must be provided
    # set path through config.yaml
    head_mesh, desired_eye_distance = load_head()
    ear_trace_mesh = o3d.geometry.PointCloud()
    test_mesh = o3d.geometry.TriangleMesh()
    
    # view head mesh in 3d mirroring user movements
    if visualizer3d:
        start_visualizer3d(head_mesh)
    # view camera feed with landmarks printed in 2d
    if visualizer2d:
        start_visualizer2d()
    
    # initialize the chosen source
    if source["type"] == "camera":
        start_webcam()
    if source["type"] == "folder":
        images_from_folder = get_images_from_folder()
    if source["type"] == "lucid-camera":
        num_channels = start_lucid_camera()

    # local variables
    frame_counter = 0
    deca_and_solvepnp_time = 0
    start_time = 0
    end_time = 0
    frame_time_current = 0
    frame_time_previous = 0
    fpss = []
    acquisition_times = []
    torch.set_grad_enabled(False)

    while True:
        frame_time_current = time.time()
        acquisition_time_start = time.time()
        # start_acquisition_memory = memory_usage(max_usage=True)

        if source["type"] == "camera":
            ret, input_image = camera.read()
            # If frame is read correctly, ret is True
            if not ret:
                myPrint("Error: Can't receive frame (stream end?). Exiting ...")
                break
        elif source["type"] == "lucid-camera":
            lucid_frame,lucid_frame_item = get_lucid_frame(num_channels)
            input_image = cv2.resize(lucid_frame, (camera_width, camera_height))
        elif source["type"] == "folder":
            if images_from_folder.__len__() > 0:
                current_image = images_from_folder.pop(0)
                input_image = current_image[0]
                input_image_name = current_image[1]
                myPrint(
                    f">>>>>> input_image {input_image_name} is {input_image.shape}, remaining {images_from_folder.__len__()}"
                )
            else:
                myPrint("All images processed (or No Input Image at all)")
                return
        frame_counter +=1
        
        images_txt.write(f"\n ---- frame {frame_counter} --- \n {input_image} ")
        images_txt.flush()
        # end_acquisition_memory = memory_usage(max_usage=True)
        # acquisition_memory = end_acquisition_memory - start_acquisition_memory
        acquisition_time_end = time.time()
        acquisition_time = acquisition_time_end - acquisition_time_start
        acquisition_times.append(acquisition_time)
        myPrint(f">>>> START FRAME {frame_counter}")
        myPrint(f"acquisition from source feed - time is {acquisition_time:.3f}")
        
        start_time = time.time()
        # start_deca_and_solvepnp_memory = memory_usage(max_usage=True)
 
        # ----------------------------
        results = None
        landmarks2Dfullres = None
        vertices = None
        bbox = None
        translation_vector = None
        rotation_vector = None
        
        results = deca_and_solvepnp(input_image, desired_eye_distance, camera_matrix,camera_dist_coeffs)
        success = results['success']
        if success:
            # landmarks3D = results['landmarks3D']
            landmarks2Dfullres = results['landmarks2Dfullres']
            vertices = results['vertices']
            bbox = results['bbox']
            translation_vector = results['translation_vector']
            rotation_vector = results['rotation_vector']
        # ----------------------------
        
        # end_deca_and_solvepnp_memory = memory_usage(max_usage=True)
        # deca_and_solvepnp_memory = end_deca_and_solvepnp_memory - start_deca_and_solvepnp_memory
        end_time = time.time()
        deca_and_solvepnp_time = end_time - start_time
        
        fps_num = (1/(frame_time_current - frame_time_previous))
        fps = f"{fps_num:.2f}"
        fpss.append(fps_num)
        avg_fps = sum(fpss) / len(fpss)

        # fps = f"{1/deca_and_solvepnp_time:.2f}"
        frame_time_previous = frame_time_current
        
        if success:
            start_visualize_time = time.time()
            # start_visualize_memory = memory_usage(max_usage=True)
            
            distance = np.linalg.norm(translation_vector)
            transform_matrix = compose_transform_matrix(rotation_vector, translation_vector)
            #save early for debug
            if save_solvepnp_data:
                save_solvepnp_transform(transform_matrix)
                save_landmarks(results, frame_counter)
            
            if visualizer3d:
                # Notice: Head_mesh will have it's vertices updated by this method
                head_mesh_rotated_translated_mirrored, ear_points_2d = visualize3d(
                    head_mesh, 
                    vertices, 
                    rotation_vector, 
                    translation_vector, 
                    camera_matrix,
                    camera_dist_coeffs)
            if visualizer2d:
                image_with_landmarks_and_bbox_flipped, image_with_landmarks_and_bbox_not_flipped = visualize2d(input_image,landmarks2Dfullres,bbox, ear_points_2d, distance, cameraFeedOnly = False)
            
            # end_visualize_memory = memory_usage(max_usage=True)
            # visualize_memory = end_visualize_memory - start_visualize_memory
            
            end_visualize_time = time.time()
            visualize_time = end_visualize_time - start_visualize_time

            if save_video_feed:
                save_feed(input_image, image_with_landmarks_and_bbox_not_flipped, frame_counter)
            if save_mesh_expression_with_landmarks3d and source["type"] == "folder":
                save_mesh_3d(head_mesh_rotated_translated_mirrored)
            if save_image_with_landmarks2d and source["type"] == "folder":
                save_img_2d(image_with_landmarks_and_bbox_not_flipped, frame_counter, input_image_name)
            if save_ear_points:
                saveEarPoints(
                    ear_trace_mesh, 
                    test_mesh, 
                    deca_vertices = vertices,
                    deca_rotation = rotation_vector, 
                    deca_translation = translation_vector)
            myPrint(
                f"Deca and SolvePNP time > {deca_and_solvepnp_time} [visualize:{visualize_time}] - dist {distance}"
            )
        else:
            myPrint(f"Image only")
            image_feed_only = visualize2d(input_image,None,None,None,None,cameraFeedOnly = True)
        # cleanup
        if source["type"] == "lucid-camera":
            cleanup_lucidcamera_buffer(lucid_frame_item)
        
        # if time_logs:
            # myPrint(f"acquisition_memory is {acquisition_memory:.3f}")
            # myPrint(f"deca_and_solvepnp_memory is {deca_and_solvepnp_memory:.3f}")
            # myPrint(f"visualize_memory is {visualize_memory:.3f}")
        
        myPrint(f"--------------------------")
        if(should_exit):
            avg_acquisition_time = sum(acquisition_times) / len(acquisition_times)
            myPrint(f"AVG acquisition from source time {avg_acquisition_time:.2f}")
            myPrint(f"AVG fps time {avg_fps:.2f}")
            return

# @profile
def deca_and_solvepnp(input_image, desired_eye_distance, camera_matrix,camera_dist_coeffs):
    # global landmarks3D, vertices, landmarks2Dfullres
    # global translation_vector, rotation_vector, bbox
    # ---- ACQUISITION
    start_acquisition_time = time.time()
    # start_acquisition_memory = memory_usage(max_usage=True)
    
    # imagedata = datasets.CameraData(input_image, face_detector = detectors.FAN(), time_logs=time_logs)[0]
    imagedata = create_camera_data(input_image)
    
    # end_acquisition_memory = memory_usage(max_usage=True)
    # acquisition_memory = end_acquisition_memory - start_acquisition_memory
    end_acquisition_time = time.time()
    acquisition_time = end_acquisition_time - start_acquisition_time
    if(imagedata == None):
        myPrint("NO FACE!")
        results = {'success': False}
        return results
        
    # ---- end ACQUISITION
    
    # ---- PREPARE DATA
    start_preparedata_time = time.time()
    # start_preparedata_memory = memory_usage(max_usage=True)

    image = imagedata["image"].to(device)[None, ...]
    tform = imagedata["tform"][None, ...]
    tform = torch.inverse(tform).transpose(1, 2).to(device)
    bbox = imagedata["bbox"]
    original_image = imagedata["original_image"][None, ...].to(device)
    _, image_height, image_width = imagedata["original_image"].shape

    # end_preparedata_memory = memory_usage(max_usage=True)
    # preparedata_memory = end_preparedata_memory - start_preparedata_memory
    end_preparedata_time = time.time()
    preparedata_time = end_preparedata_time - start_preparedata_time
    # ---- end PREPARE DATA

    # ---- ENCODING ----
    start_encoding_time = time.time()
    # start_encoding_memory = memory_usage(max_usage=True)

    codedict = deca.encode(image)

    # end_encoding_memory = memory_usage(max_usage=True)
    # encoding_memory = end_encoding_memory - start_encoding_memory
    end_encoding_time = time.time()
    encoding_time = end_encoding_time - start_encoding_time
    
    # ---- DECODING ----
    start_decoding_time = time.time()
    # start_decoding_memory = memory_usage(max_usage=True)

    opdict = deca.decode_fast(
        codedict,
        original_image=original_image,
        tform=tform,
        full_res_landmarks2D=True,
    )
    
    # end_decoding_memory = memory_usage(max_usage=True)
    # decoding_memory = end_decoding_memory - start_decoding_memory
    end_decoding_time = time.time()
    decoding_time = end_decoding_time - start_decoding_time
    
    # ---- LANDMARKS ----
    start_landmarks_time = time.time()
    # start_landmarks_memory = memory_usage(max_usage=True)

    landmarks3D = opdict["landmarks3d_world"][0].cpu().numpy()[:, :3]
    landmarks3D, _ = scalePoints(landmarks3D, 39, 42, desired_eye_distance)
    landmarks3D = np.ascontiguousarray(landmarks3D, dtype=np.float32)
    eye_distance_now1 = getDesiredEyeDistance(landmarks3D, 39, 42)
    myPrint(f"Landmarks3d : After scaling to reach {desired_eye_distance}, distance is: {eye_distance_now1}")

    landmarks2Dfullres = opdict["landmarks2d_full_res"][0]
    landmarks2Dfullres = np.ascontiguousarray(landmarks2Dfullres, dtype=np.float32)

    vertices = opdict["verts"][0].cpu().numpy()[:, :3]
    # TODO: shouldn't we enforce that the distance between 3827, 3619
    # is indeed the desired_eye_distance, instead of applying the same
    # scaling factor we applied to the landmakark3D ?
    vertices, _ = scalePoints(vertices, 3827, 3619, desired_eye_distance)
    
    # scalePoints(vertices, desiredScalingFactor=scaling_factor)
    eye_distance_now = getDesiredEyeDistance(vertices, 3827, 3619)
    myPrint(f"Vertices: After scaling to reach {desired_eye_distance}, distance is: {eye_distance_now}")
    
    # end_landmarks_memory = memory_usage(max_usage=True)
    # landmarks_memory = end_landmarks_memory - start_landmarks_memory
    end_landmarks_time = time.time()
    landmarks_time = end_landmarks_time - start_landmarks_time
    
    # ---- SOLVEPNP ----
    start_solvepnp_time = time.time()
    # start_solvepnp_memory = memory_usage(max_usage=True)

    success, rotation_vector, translation_vector = cv2.solvePnP(
        landmarks3D,
        # landmarks2D,
        landmarks2Dfullres,
        camera_matrix,
        camera_dist_coeffs,
    )
    
    # end_solvepnp_memory = memory_usage(max_usage=True)
    # solvepnp_memory = end_solvepnp_memory - start_solvepnp_memory
    end_solvepnp_time = time.time()
    solvepnp_time = end_solvepnp_time - start_solvepnp_time

    #PRINT:
    if time_logs:
        total_time = acquisition_time + preparedata_time + encoding_time + decoding_time + landmarks_time + solvepnp_time
        acquisition_time_percentage = getPercentage(acquisition_time, total_time)
        preparedata_time_percentage = getPercentage(preparedata_time, total_time)
        encoding_time_percentage = getPercentage(encoding_time, total_time)
        decoding_time_percentage = getPercentage(decoding_time, total_time)
        landmarks_time_percentage = getPercentage(landmarks_time, total_time)
        solvepnp_time_percentage = getPercentage(solvepnp_time, total_time)
        myPrint(f"--- TIMERS ---[{total_time:.3f}]")
        myPrint(f"bbox and transform  [{acquisition_time_percentage:.1f}%] {acquisition_time:.3f}")
        myPrint(f"preparedata_time [{preparedata_time_percentage:.1f}%] {preparedata_time:.3f}")
        myPrint(f"encoding_time [{encoding_time_percentage:.1f}%] {encoding_time:.3f}")
        myPrint(f"decoding_time [{decoding_time_percentage:.1f}%] {decoding_time:.3f}")
        myPrint(f"landmarks_time [{landmarks_time_percentage:.1f}%] {landmarks_time:.3f}")
        myPrint(f"solvepnp_time [{solvepnp_time_percentage:.1f}%] {solvepnp_time:.3f}")
        # myPrint(f"--- MEMORY ---[{total_time:.3f}]")
        # myPrint(f"acquisition_memory {acquisition_memory:.3f}")
        # myPrint(f"preparedata_memory {preparedata_memory:.3f}")
        # myPrint(f"encoding_memory {encoding_memory:.3f}")
        # myPrint(f"decoding_memory {decoding_memory:.3f}")
        # myPrint(f"landmarks_memory {landmarks_memory:.3f}")
        # myPrint(f"solvepnp_memory {solvepnp_memory:.3f}")
    
    results = {
        'success': success,
        'landmarks3D': landmarks3D,
        'landmarks2Dfullres': landmarks2Dfullres,
        'vertices': vertices,
        'bbox': bbox,
        'translation_vector': translation_vector,
        'rotation_vector': rotation_vector
    }
    return results

def getPercentage(part,whole):
    return part/whole * 100

def visualize3d(head_mesh, deca_vertices, deca_rotation, deca_translation, camera_matrix,camera_dist_coeffs):

    vertices_rotated = applyRotation(deca_vertices,deca_rotation, mirror=True)
    vertices_rotated_translated = applyTranslation(vertices_rotated, deca_translation, mirror=True)
    head_mesh.vertices = o3d.utility.Vector3dVector(vertices_rotated_translated)

    visualizer3d.update_geometry(head_mesh)
    # We need the rotated translated mirrored head mesh, to grab the earpoints
    ear_points_2d = projectEarPointsTo2D(head_mesh, camera_matrix, camera_dist_coeffs)
    visualizer3d.poll_events()
    visualizer3d.update_renderer()
    return head_mesh, ear_points_2d

def applyRotation(input_vertices, input_rotation_vector, mirror=True): 
    if SAFE_COPY:
        vertices = np.copy(input_vertices)
        rotation_vector = np.copy(input_rotation_vector)
    else:
        vertices = input_vertices
        rotation_vector = input_rotation_vector
    rotation_matrix = cv2.Rodrigues(rotation_vector)[0].T
    if mirror:
        rotation_matrix[0, :] *= 1  # Invert X-axis
        rotation_matrix[1, :] *= -1  # Invert Y-axis
        rotation_matrix[2, :] *= -1  # Invert Z-axis
    return np.matmul(vertices, rotation_matrix)

def applyTranslation(input_vertices,input_translation_vector, mirror=True):
    if SAFE_COPY:
        vertices = np.copy(input_vertices)
        translation = np.copy(input_translation_vector)
    else:
        vertices = input_vertices
        translation = input_translation_vector

    if mirror:
        translation *= tmirror 
    return vertices + translation.T

def visualize2d(input_image,landmarks2Dfullres,bbox, ear_points_2d,distance, cameraFeedOnly = False):
    if SAFE_COPY:
        image = np.copy(input_image)
    else:
        image = input_image
        
    if cameraFeedOnly:
        myPrint("showing feed only")
        flipped_image = cv2.flip(image, 1)
        cv2.imshow(camera_window_name, flipped_image)
        cv2.waitKey(1)
        return;
        
    # landmarksFlipped = cv2.flip(landmarks2Dfullres, 1)
    # myPrint(f"Visualize2d landmarks2Dfullres.shape {landmarks2Dfullres.shape} -  input_image.shape {input_image.shape}")
    # input_image = draw_points(input_image, landmarks2Dfullres)

    image_with_landmarks = draw_points(image, landmarks2Dfullres)
    
    image_with_landmarks_and_ears = draw_points(image_with_landmarks, ear_points_2d, 3, (255,0,0))
    image_with_bbox = draw_rect(image_with_landmarks_and_ears, bbox)
    
    flipped_image_with_bbox = cv2.flip(image_with_bbox, 1)

    text = f"Dist. cm: {distance:.3f} \n Time s:{deca_and_solvepnp_time:.3f}"
    flipped_image_with_bbox = draw_text(flipped_image_with_bbox, text, "top-right")
    myPrint(f"fps {fps}")
    flipped_image_with_bbox = draw_text(flipped_image_with_bbox, fps, "top-left")

    myPrint(f"Visualize2d input_image.shape {flipped_image_with_bbox.shape}")
    cv2.imshow(camera_window_name, flipped_image_with_bbox)
    return image_with_bbox, flipped_image_with_bbox

# def save_video(frame_counter):
#     original_frames_path = os.path.join(output_folder, "original_frames", f"original_frame_{frame_counter}.png")
#     cv2.imwrite(original_frames_path, original_input_image)
    
#     # deca_frames_path = os.path.join(output_folder, "deca_frames", f"deca_frame_{image_counter}.png")
    
def projectEarPointsTo2D(translated_rotated_mirrored_head_mesh, camera_matrix, camera_dist_coeffs):

    ear_points = getEarPoints3D(translated_rotated_mirrored_head_mesh, None, decaReferenceSystem = False);
    
    ear_points_2d, _= cv2.projectPoints(ear_points, np.eye(3), np.zeros(3), camera_matrix, camera_dist_coeffs)
    ear_points_2d = ear_points_2d.squeeze()
    return ear_points_2d
    
def getEarPoints3D(translated_rotated_mirrored_head_mesh, deca_vertices, decaReferenceSystem = False):
    right_ear_index = 1760
    left_ear_index = 502
    
    if(decaReferenceSystem):
        if SAFE_COPY:
            vertices = np.copy(deca_vertices)
        else: 
            vertices = deca_vertices
        # vertices contains the result of DECA
        desired_eye_distance = getDesiredEyeDistance(vertices, 3827, 3619)
        myPrint(f"vertices now with eye distance at {desired_eye_distance}")
        head_mesh_vertices = np.asarray(vertices)
    else:
        #head_mesh at this point has been rotated and mirrored
        # for 3d visualization
        head_mesh_vertices = np.asarray(translated_rotated_mirrored_head_mesh.vertices)
    ear_points_3d = np.array([head_mesh_vertices[left_ear_index],head_mesh_vertices[right_ear_index]])
    return ear_points_3d
       
def draw_text(input_image, text, position="top-right"):
    if SAFE_COPY:
        image = np.copy(input_image)
    else:
        image = input_image
    # Define the font for the text
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Define the starting position for the text (top right corner)
    if position == "top-right":
        base_x = image.shape[1] - 280
        base_y = 40
    elif position == "top-left":
        base_x = 40
        base_y = 40
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
        cv2.putText(image, 
                    line,  # Text string for the current line
                    line_position, 
                    font, 
                    font_scale, 
                    font_color, 
                    line_type)
    return image
def save_mesh_3d(head_mesh):
    now = datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")
    relative_path = os.path.join(
        # script_dir,
        output_folder,
        f"mesh_expression_{date_time}.ply",
    )
    # head_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d.io.write_triangle_mesh(
        relative_path,
        head_mesh,
    )

def save_img_2d(input_image, frame_counter, input_image_name = None):
    if SAFE_COPY:
        image = np.copy(input_image)
    else:
        image = input_image
    # now = datetime.now()
    # This will give you time precise up to milliseconds
    # date_time = now.strftime("%Y%m%d_%H%M%S%f")[:17]  
    if input_image_name is None:
        image_path = os.path.join(output_folder,f"output_image_{frame_counter}.png")
    else:
        image_path = os.path.join(output_folder,f"output_image_{input_image_name}_{frame_counter}.png")
        
    # relative_path_original = os.path.join(output_folder,f"ORIGINAL_image_landmarks2d_{date_time}.png")
    # myPrint(f"saving {date_time}")
    cv2.imwrite(image_path, image)
    # cv2.imwrite(relative_path_original, image_copy)
    # frame_with_landmarks = draw_points(image_copy, landmarks2Dfullres)
    # cv2.imwrite(relative_path, frame_with_landmarks)

def save_feed(input_original_image, input_processed_image, frame_counter):
    if SAFE_COPY:
        original_image = np.copy(input_original_image)
        processed_image = np.copy(input_processed_image)
    else:
        original_image = input_original_image
        processed_image = input_processed_image
        
    original_image_path = os.path.join(output_folder,f"frame_{frame_counter}_original.png")
    processed_image_path = os.path.join(output_folder,f"frame_{frame_counter}_processed.png")
    
    # Attention:
    # if i save the original image flipped, it will be better when 
    # reviewing the results, but when i use the saved originals 
    # to run the folder mode, they will be inherently different 
    # from the true original upon which the webcame mode did the face detection.
    # therefore, it's better to save the original "as seen by the camera" 
    # which is what the deca+solvepnp will get, so that both processed are really the same
    # and thus comparable

    # original_image_flipped = cv2.flip(original_image, 1)
    # cv2.imwrite(original_image_path, original_image_flipped)
    cv2.imwrite(original_image_path, original_image)
    cv2.imwrite(processed_image_path, processed_image)

def compose_transform_matrix(input_rotation_vector, input_translation_vector):
    transform_matrix = np.eye(4)
    if SAFE_COPY:
        rotation_vector = np.copy(input_rotation_vector)
        translation_vector = np.copy(input_translation_vector[0])
    else:
        rotation_vector = input_rotation_vector
        translation_vector = input_translation_vector[0]
    transform_matrix[:3, :3] = cv2.Rodrigues(rotation_vector)[0].T
    transform_matrix[:3, 3] = translation_vector.T
    return transform_matrix
def save_landmarks(results, frame_counter):
    landmarks2Dfullres = results['landmarks2Dfullres']
    vertices = results['vertices']
    bbox = results['bbox']
    translation_vector = results['translation_vector']
    rotation_vector = results['rotation_vector']
    with open(output_folder+f"/landmarks2Dfullres.txt", 'a') as file:
        file.write(f"\n -----{frame_counter}----- \n")
        np.savetxt(file, landmarks2Dfullres, fmt='%.4f')
    with open(output_folder+f"/vertices.txt", 'a') as file:
        file.write(f"\n -----{frame_counter}----- \n")
        np.savetxt(file, vertices, fmt='%.4f')
    with open(output_folder+f"/bbox.txt", 'a') as file:
        file.write(f"\n -----{frame_counter}----- \n")
        np.savetxt(file, bbox, fmt='%.4f')
    with open(output_folder+f"/translation_vector.txt", 'a') as file:
        file.write(f"\n -----{frame_counter}----- \n")
        np.savetxt(file, translation_vector, fmt='%.4f')
    with open(output_folder+f"/rotation_vector.txt", 'a') as file:
        file.write(f"\n -----{frame_counter}----- \n")
        np.savetxt(file, rotation_vector, fmt='%.4f')
    
     
def save_solvepnp_transform(transform_matrix):
    with open(output_folder+f"/solvepnp_transform.txt", 'a') as file:
        file.write(f"\n -------- \n")
        np.savetxt(file, transform_matrix)

def saveEarPoints(ear_trace_mesh, test_mesh, deca_vertices,deca_rotation, deca_translation):
    new_ear_points = getEarPoints3D(None, deca_vertices, decaReferenceSystem = True)
    ear_points_rotated = applyRotation(new_ear_points, deca_rotation, mirror=False)
    ear_points_translated = applyTranslation(ear_points_rotated,deca_translation, mirror=False)
    timestamp = time.time()
    for ear_point, label in zip(ear_points_translated, ["leftEar", "rightEar"]):
        row = [timestamp, *ear_point, label]
        csv_writer.writerow(row)
        # Flush the contents to the file to ensure it's written immediately
        ear_csv.flush()
    
    new_ear_points = np.vstack((np.asarray(ear_trace_mesh.points), ear_points_translated))
    ear_trace_mesh.points = o3d.utility.Vector3dVector(new_ear_points)
    path_ply = os.path.join(output_folder, f"ear_trace.ply")   
    o3d.io.write_point_cloud(path_ply, ear_trace_mesh)
    
    # test save head mesh too
    vertices_rotated = applyRotation(deca_vertices, deca_rotation, mirror=False)
    vertices_translated = applyTranslation(vertices_rotated,deca_translation, mirror=False)
    # new_vertices = np.vstack((np.asarray(test_mesh.vertices), vertices_translated))
    # test_mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
    test_mesh.vertices = o3d.utility.Vector3dVector(vertices_translated)
    relative_path2 = os.path.join(output_folder,f"vertices_test.ply")
    o3d.io.write_triangle_mesh(relative_path2, test_mesh)
    
    with open(output_folder+f"/ear_points.txt", 'ab') as file:
        np.savetxt(file, new_ear_points)
        
def draw_rect(input_image, bbox):
    if SAFE_COPY:
        image = np.copy(input_image)
    else:
        image = input_image

    left = int(round(bbox[0]))
    top = int(round(bbox[1]))
    right = int(round(bbox[2]))
    bottom = int(round(bbox[3]))
    cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)  # (B, G, R) color values
    return image   
def draw_points(input_image, input_point_list, radius=2, color=(0, 0, 255)):
    if SAFE_COPY:
        image = np.copy(input_image)
        point_list = np.copy(input_point_list)
    else:
        image = input_image    
        point_list = input_point_list

    for point in point_list:
        # myPrint(f"draw_points of point {point}")
        x, y = int(round(point[0])), int(round(point[1]))
        cv2.circle(image, (x, y), radius=radius, color=color, thickness=-1)
    return image
def getDesiredEyeDistance(vertices, fromPointIndex, toPointIndex):
    # this will be the desired size to which scale the landmarks
    return math.dist(vertices[fromPointIndex], vertices[toPointIndex])

def scalePoints(
    input_points,
    fromPointIndex=None,
    toPointIndex=None,
    desiredDistance=None,
    desiredScalingFactor=None,
):
    if SAFE_COPY:
        points = np.copy(input_points)
    else:
        points = input_points

    if fromPointIndex != None and toPointIndex != None:
        currentEyeDistance = math.dist(points[fromPointIndex], points[toPointIndex])
        scalingFactor = desiredDistance / currentEyeDistance
        # myPrint(f"SCALING - initial distance was {currentEyeDistance} - desired is {desiredDistance} : scalingFactor {scalingFactor}")
        points *= scalingFactor
    elif desiredScalingFactor != None:
        points *= desiredScalingFactor
        scalingFactor = desiredScalingFactor
    return points, scalingFactor

def add_gizmo(vis, origin=None, size=1):
    # Create a coordinate frame (gizmo) at the center of the scene
    if origin is None:
        origin = [0,0,0]
    gizmo = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=origin)
    vis.add_geometry(gizmo)
def add_wireframes(vis):
    commonWidth = 200
    commonHeight = 200
    commonDepth = 200
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
def start_visualizer3d(head_mesh):
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
    # visualizer3d.register_key_callback(ord("P"), lambda vis, key: on_press_p(image_to_save_on_q_press,landamarks2d_to_save_on_q, head_mesh_to_save))

def load_head():
    # global head_mesh, desired_eye_distance
    head_mesh = o3d.io.read_triangle_mesh(head_mesh_path)
    vertices = np.asarray(head_mesh.vertices)
    # head mesh is already perfect scale. we save eye distance
    # to scale landmarks later on
    desired_eye_distance = getDesiredEyeDistance(vertices, 3827, 3619)
    myPrint(f"Desired Eye distance is {desired_eye_distance}")
    return head_mesh, desired_eye_distance
def start_webcam():
    global camera

    camera = cv2.VideoCapture(camera_index)

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)

    if not camera.isOpened():
        myPrint("Error: Could not open video stream.")
        exit()
def start_lucid_camera():
    global lucidCamera
    devices = create_devices_with_tries()
    lucidCamera = devices[0]
    # Setup
    nodemap = lucidCamera.nodemap
    nodes = nodemap.get_node(['Width', 'Height', 'PixelFormat', 'DecimationHorizontal', 'DecimationVertical'])

    # nodes['Width'].value = camera_width
    # nodes['Height'].value = camera_height
    nodes['DecimationHorizontal'].value = 1
    nodes['Width'].value = 2048
    nodes['Height'].value = 1536    
    nodes['PixelFormat'].value = 'BGR8'
    nodes['DecimationHorizontal'].value = 2
    # nodes['DecimationVertical'].value = 2
    num_channels = 3

    # Stream nodemap
    tl_stream_nodemap = lucidCamera.tl_stream_nodemap

    tl_stream_nodemap["StreamBufferHandlingMode"].value = "NewestOnly"
    tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = True
    tl_stream_nodemap['StreamPacketResendEnable'].value = True
    
    lucidCamera.start_stream()
    return num_channels
def create_devices_with_tries():
	'''
	This function waits for the user to connect a device before raising
		an exception
	'''

	tries = 0
	tries_max = 6
	sleep_time_secs = 10
	while tries < tries_max:  # Wait for device for 60 seconds
		devices = system.create_device()
		if not devices:
			myPrint(
				f'Try {tries+1} of {tries_max}: waiting for {sleep_time_secs} '
				f'secs for a device to be connected!')
			for sec_count in range(sleep_time_secs):
				time.sleep(1)
				myPrint(f'{sec_count + 1 } seconds passed ',
					'.' * sec_count, end='\r')
			tries += 1
		else:
			myPrint(f'Created {len(devices)} device(s)')
			return devices
	else:
		raise Exception(f'No device found! Please connect a device and run '
						f'the example again.')
def get_lucid_frame(num_channels):
    buffer = lucidCamera.get_buffer()
    """
    Copy buffer and requeue to avoid running out of buffers
    """
    lucid_frame_item = BufferFactory.copy(buffer)
    lucidCamera.requeue_buffer(buffer)

    buffer_bytes_per_pixel = int(len(lucid_frame_item.data)/(lucid_frame_item.width * lucid_frame_item.height))
    """
    Buffer data as cpointers can be accessed using buffer.pbytes
    """
    array = (ctypes.c_ubyte * num_channels * lucid_frame_item.width * lucid_frame_item.height).from_address(ctypes.addressof(lucid_frame_item.pbytes))
    """
    Create a reshaped NumPy array to display using OpenCV
    """
    npndarray = np.ndarray(buffer=array, dtype=np.uint8, shape=(lucid_frame_item.height, lucid_frame_item.width, buffer_bytes_per_pixel))
            
    return npndarray, lucid_frame_item
def cleanup_lucidcamera_buffer(lucid_frame_item):
    """
    Destroy the copied item to prevent memory leaks
    NOTE: do this after any draw/show which use the item
        and any derived data like array and npndarray
    """
    BufferFactory.destroy(lucid_frame_item)

def get_images_from_folder():
    valid_extensions = (".jpg", ".jpeg", ".png")
    all_files = os.listdir(image_folder)
    image_names = [
        file for file in all_files if file.lower().endswith(valid_extensions)
    ]
    images = [
        (cv2.imread(os.path.join(image_folder, image_name)), image_name)
       for image_name in image_names
    ]
    myPrint(f">>>>> got images {images.__len__()}")
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
def stop_lucid_camera():
    lucidCamera.stop_stream()
    nodemap = lucidCamera.nodemap
    nodes = nodemap.get_node(['Width', 'Height', 'PixelFormat', 'DecimationHorizontal', 'DecimationVertical'])

    # nodes['Width'].value = camera_width
    # nodes['Height'].value = camera_height
    nodes['DecimationHorizontal'].value = 1
    nodes['Width'].value = 2048
    nodes['Height'].value = 1536    
    nodes['PixelFormat'].value = 'BGR8'
    # nodes['DecimationVertical'].value = 2
    system.destroy_device()

def on_press_q(e):
    global should_exit
    myPrint("KEY q")
    should_exit = True
    stop_visualizer3d()
    if(source["type"] == "lucid-camera"):
        stop_lucid_camera()
    elif(source["type"] == "camera"):
        stop_webcam()
        
# def on_press_p(image_to_save,landmarks2d_to_save, head_mesh_to_save):
#     myPrint("KEY p")
#     save_mesh_3d(head_mesh_to_save)
#     save_img_2d(image_to_save,landmarks2d_to_save)

def create_camera_data(input_image):
    crop_size=224 
    # crop_size=192 

    # Prepare Image
    start_prepareimage_time = time.time()
    # ------
    image = prepare_image(input_image)
    # ------    
    end_prepareimage_time = time.time()
    prepareimage_time = end_prepareimage_time - start_prepareimage_time
    
    # Detect Face
    start_facedetector_time = time.time()
    # ------
    bbox_type = None
    bbox = None
    left = None
    right = None
    top = None
    bottom = None
    results = None
    results = detectFace(image)
    if results == None:
        return None
    else: bbox_type, bbox, left, right, top, bottom = results
    # bbox_type, bbox, left, right, top, bottom = detectFace(image)
    # ------
    end_facedetector_time = time.time()
    facedetector_time = end_facedetector_time - start_facedetector_time
    
    # Calc the src points
    start_bbox2point_time = time.time()
    # ------
    src_pts = bbox2pointCalc(bbox_type, left, right, top, bottom)
    # ------
    end_bbox2point_time = time.time()
    bbox2point_time = end_bbox2point_time - start_bbox2point_time
    
    # Calc the transform
    start_estimatetransform_time = time.time()
    # ------
    tform = estimateTransformCalc(src_pts, crop_size)
    # ------
    end_estimatetransform_time = time.time()
    estimatetransform_time = end_estimatetransform_time - start_estimatetransform_time

    # Warp
    start_warp_time = time.time()
    # ------
    dst_image = warpCalc(image, tform, crop_size)
    # ------
    end_warp_time = time.time()
    warp_time = end_warp_time - start_warp_time
    
    # Prepare tensors
    start_tensors_time = time.time()
    # ------
    image_tensor = torch.tensor(dst_image).float()
    tform_tensor = torch.tensor(tform.params).float()
    original_image_tensor = torch.tensor(image.transpose(2,0,1)).float()
    # ------

    end_tensors_time = time.time()
    tensors_time = end_tensors_time - start_tensors_time
    
    if(time_logs):
            total_time = prepareimage_time + facedetector_time + bbox2point_time + estimatetransform_time + warp_time + tensors_time
            prepareimage_time_percentage = getPercentage(prepareimage_time, total_time)
            facedetector_time_percentage = getPercentage(facedetector_time, total_time)
            bbox2point_time_percentage = getPercentage(bbox2point_time, total_time)
            estimatetransform_time_percentage = getPercentage(estimatetransform_time, total_time)
            warp_time_percentage = getPercentage(warp_time, total_time)
            tensors_time_percentage = getPercentage(tensors_time, total_time)
            myPrint(f"++++++++ datasets TIMERS ---[{total_time:.3f}]")
            myPrint(f"++++++++ prepareimage_time [{prepareimage_time_percentage:.1f}%] {prepareimage_time:.3f}")
            myPrint(f"++++++++ facedetector_time [{facedetector_time_percentage:.1f}%] {facedetector_time:.3f}")
            myPrint(f"++++++++ bbox2point_time [{bbox2point_time_percentage:.1f}%] {bbox2point_time:.3f}")
            myPrint(f"++++++++ estimatetransform_time [{estimatetransform_time_percentage:.1f}%] {estimatetransform_time:.3f}")
            myPrint(f"++++++++ warp_time [{warp_time_percentage:.1f}%] {warp_time:.3f}")
            myPrint(f"++++++++ tensors_time [{tensors_time_percentage:.1f}%] {tensors_time:.3f}")   
            myPrint(f"++++++++++++")
     
    return {'image': image_tensor,
            'imagename': 'frame',
            'tform': tform_tensor,
            'original_image': original_image_tensor,
            'bbox': bbox,
            'bbox2point': bbox2point}   

def prepare_image(input_image):
    if SAFE_COPY:
        image = np.array(np.copy(input_image))
    else:
        image = np.array(image)

    if len(image.shape) == 2:
        image = image[:,:,None].repeat(1,1,3)
    if len(image.shape) == 3 and image.shape[2] > 3:
        image = image[:,:,:3]
    
    return image
def detectFace(input_image):
    if SAFE_COPY:
        image = np.array(np.copy(input_image))
    else:
        image = np.array(image)

    # bbox, bbox_type = mediaPipeFaceMeshDetectionOriginal(image)
    bbox, bbox_type = mediaPipeFaceMeshDetection(image)
    # bbox, bbox_type = mediaPipeFaceDetection(image)
    if bbox is None:
        myPrint('no face detected! return null')
        return None
    else:
        left = bbox[0]; right=bbox[2]
        top = bbox[1]; bottom=bbox[3]
    return bbox_type, bbox, left, right, top, bottom             
def bbox2pointCalc(bbox_type, left, right, top, bottom):  
    scale=1.25
    old_size, center = bbox2point(left, right, top, bottom, type=bbox_type)
    size = int(old_size*scale)
    src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
    return src_pts    
def bbox2point(left, right, top, bottom, type='bbox'):
        ''' bbox from detector and landmarks are different
        '''
        if type=='kpt68':
            old_size = (right - left + bottom - top)/2*1.1
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
        elif type=='bbox':
            old_size = (right - left + bottom - top)/2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0  + old_size*0.12])
        else:
            raise NotImplementedError
        return old_size, center
def estimateTransformCalc(src_pts,crop_size):
    DST_PTS = np.array([[0,0], [0,crop_size - 1], [crop_size - 1, 0]])
    tform = estimate_transform('similarity', src_pts, DST_PTS)
    return tform
def warpCalc(image,tform, crop_size):
    image = image/255.
    dst_image = warp(image, tform.inverse, output_shape=(crop_size, crop_size))
    dst_image = dst_image.transpose(2,0,1)
    return dst_image    
def getPercentage(part,whole):
    try:
        result = part/whole * 100
        return result
    except ZeroDivisionError:
        return 0
def mediaPipeFaceMeshDetectionOriginal(image):
    # mp_face_mesh = mp.solutions.face_mesh
    # model = mp_face_mesh.FaceMesh()
    out = model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # out = self.model.get_landmarks(image)
    if out.multi_face_landmarks is None:
        out = None
        return None, None
    else:
        face_landmarks = out.multi_face_landmarks[0]
        kpt = np.array([(landmark.x, landmark.y) 
                        for landmark in face_landmarks.landmark])
        # Convert normalized coordinates to pixel values
        kpt[:, 0] *= image.shape[1]  # Multiply x by image width
        kpt[:, 1] *= image.shape[0]  # Multiply y by image height
        # Compute the bounding box coordinates
        left = np.min(kpt[:, 0])
        right = np.max(kpt[:, 0])
        top = np.min(kpt[:, 1])
        bottom = np.max(kpt[:, 1])
        bbox = [left, top, right, bottom]
        return bbox, 'kpt68'
    
def mediaPipeFaceMeshDetection(image):
    # try using code from Mario's implementation

    # model = mp_face_mesh.FaceMesh(
    #     max_num_faces=1,
    #     refine_landmarks=True,
    #     min_detection_confidence=0.5,
    #     min_tracking_confidence=0.5
    #     )
    h,w = image.shape[:2]
    # out = model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    out = model.process(image[:,:,::-1])
    if out.multi_face_landmarks:
        face_landmarks = out.multi_face_landmarks[0]
        pts = np.array([(pt.x * w, pt.y * h)
                        for pt in face_landmarks.landmark],
                       dtype=np.float64)
        min_coords = pts.min(axis=0)
        max_coords = pts.max(axis=0)
        bbox = [
            min_coords[0],  # left
            min_coords[1],  # top
            max_coords[0],  # right
            max_coords[1]   # bottom
        ]
        bbox = np.round(bbox).astype(np.int32)
        return bbox, 'kpt68'
    else:
        out = None
        return None, None

def mediaPipeFaceDetection(image):
    # try using face detector instead of face mesh
    h,w = image.shape[:2]
    out = face_detection.process(image)
    if out.detections:
        face_detected = out.detections[0]
        found_bbox = face_detected.location_data.relative_bounding_box
        xmin = int(found_bbox.xmin * w)
        ymin = int(found_bbox.ymin * h)
        width = int(found_bbox.width * w)
        height = int(found_bbox.height * h)
        bbox = [xmin, ymin, xmin + width, ymin + height]
        return bbox, 'kpt68'
    else:
        out = None
        return None, None
        
def load_config():
    global camera_matrix, camera_dist_coeffs
    global camera_width, camera_height
    global source, camera_index, image_folder
    global output_folder, head_mesh_path
    global visualizer2d, visualizer3d
    global visualizer2d_height, visualizer2d_width
    global save_mesh_expression_with_landmarks3d
    global save_image_with_landmarks2d
    global save_solvepnp_data
    global save_video_feed
    global save_ear_points, csv_writer, ear_csv
    global time_logs, logs_txt,images_txt
    
        
    with open(global_args.config, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return
        
    output = config["output"]    
    output_folder = os.path.join(script_dir, output["folder"])
    # Check if the directory exists, if not, create it
    os.makedirs(output_folder, exist_ok=True)
    
    # TODO: add config to control this
    # save_logs = output["save_logs"]
    save_logs = True
    if save_logs:
        logs_txt = open(output_folder+f"/logs.txt", 'a')
        images_txt = open(output_folder+f"/images.txt", 'a')
        print(f"saving logs to {logs_txt}")
        print("writing")
        logs_txt.write(f"\n -----LOGS----- \n")
        print("done")

    camera = config["camera"]
    camera_width = camera["capture_resolution_width"]
    camera_height = camera["capture_resolution_height"]
    camera_calibration_data = camera["calibration_data"]
    if camera_calibration_data != None or camera_calibration_data != "":
        camera_data = np.load(os.path.join(script_dir, camera_calibration_data))
        if camera_data != None:
            camera_matrix = np.asarray(camera_data["camera_matrix"], dtype=np.float32)
            camera_dist_coeffs = np.asarray(
                camera_data["dist_coeffs"], dtype=np.float32
            )
        else:
            myPrint("ERROR fetching camera data")
            return
    else:
        myPrint("ERROR no camera data specified")
        return
    myPrint(
        f"\n LOADED camera_matrix {camera_calibration_data} - "
        f"\n camera_matrix {camera_matrix} "
        f"\n dist_coeffs {camera_dist_coeffs} "
        f"\n w:{camera_width} h:{camera_height}"
)
    source = config["source"]
    if source["type"] == "camera":
        camera_index = source["camera_index"]
    elif source["type"] == "folder":
        image_folder = os.path.join(script_dir, source["folder"])


    
    save_mesh_expression_with_landmarks3d = output["save_mesh_expression_with_landmarks3d"]
    save_image_with_landmarks2d = output["save_image_with_landmarks2d"]
    save_solvepnp_data = output["save_solvepnp_data"]
    save_video_feed = output["save_video_feed"]
    save_ear_points = output["save_ear_points"]
    if save_ear_points:
        csv_path = os.path.join(output_folder, f"ear_trace.csv")   
        ear_csv = open(csv_path, mode='w', newline='')
        csv_writer = csv.writer(ear_csv)
        # header
        header = ["timestamp", "x", "y", "z", "label"]
        csv_writer.writerow(header)
        ear_csv.flush()
    time_logs = output["time_logs"]
    
    head_mesh_path = os.path.join(script_dir, config["head_mesh"])

    visualize = config["visualize"]
    visualizer2d = visualize["visualizer2d"]
    visualizer2d_width = visualize["visualizer2d_width"]
    visualizer2d_height = visualize["visualizer2d_height"]
    visualizer3d = visualize["visualizer3d"]
    


def myPrint(txt):
    #TODO make write optional wrt to config
    logs_txt.write(f"\n{txt}")
    logs_txt.flush()
    print(f"{txt}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DECA + SolvePnP")

    # Additional Arguments not in original lib
    parser.add_argument(
        "--config",
        default=os.path.join(script_dir,"./config.yaml"),
        type=str,
        help="path to the config file",
    )

    main(parser.parse_args())