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
from tkinter import END
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


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA

from decalib.datasets import datasets 
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg
from decalib.utils.tensor_cropper import transform_points


# GLOBAL VARIABLES
global_args = None
head_mesh = None


def main(args):
    global global_args
    global_args = args
  
    savefolder = args.savefolder
    device = args.device
    os.makedirs(savefolder, exist_ok=True)
    # load test images 
    testdata = datasets.TestData(args.inputpath, iscrop=args.iscrop, face_detector=args.detector, sample_step=args.sample_step)

    # Load camera matrix
    calibration_data = args.calibrationData
    data = np.load(calibration_data)
    camera_matrix = data['camera_matrix']
    dist_coeffs = data['dist_coeffs']
    
    # Ensure that camera_matrix and dist_coeffs are also float32 and have the correct shapes
    camera_matrix = np.asarray(camera_matrix, dtype=np.float32)
    dist_coeffs = np.asarray(dist_coeffs, dtype=np.float32)

    # print('camera_matrix', camera_matrix, '\n dist_coeffs', dist_coeffs)

    # run DECA

    # whether to use FLAME texture model to generate uv texture map
    deca_cfg.model.use_tex = args.useTex

    # pytorch3d or standard(default)
    deca_cfg.rasterizer_type = args.rasterizer_type
    # whether to extract texture from input image as the uv texture map, set false if you want albedo map from FLAME model
    deca_cfg.model.extract_tex = args.extractTex
    
    deca = DECA(config = deca_cfg, device=device)
    
    # track various execution times
    total_encode_time = 0
    total_decode_time = 0
    total_landmark_time = 0
    total_solvepnp_time = 0
    
    encoding_times = []
    decoding_times = []
    landmark_times = []
    solvepnp_times = []
    iteration_times = []
    get_image_times = []
    prepare_times = []
    prepare_image_times = []
    prepare_tform_times = []
    prepare_origimage_times = []
    
    # results
    distances = []
    all_landmarks3D = np.empty((len(testdata), 68, 3))
    all_landmarks2D = np.empty((len(testdata), 68, 2))
    
    for i in tqdm(range(len(testdata))):
        start_prepare = time.time()
        
        start_get_image = time.time()
        current_image = testdata[i]
        end_get_image = time.time()
        get_image_time = end_get_image - start_get_image
        get_image_times.append(get_image_time)
        
        start_prepare_image = time.time()
        name = current_image['imagename']
        image = current_image['image'].to(device)[None,...]
        end_prepare_image = time.time()
        prepare_image_time = end_prepare_image - start_prepare_image
        prepare_image_times.append(prepare_image_time)
        
        start_prepare_tform = time.time()
        tform = current_image['tform'][None, ...]
        tform = torch.inverse(tform).transpose(1,2).to(device)
        end_prepare_tform = time.time()
        prepare_tform_time = end_prepare_tform - start_prepare_tform
        prepare_tform_times.append(prepare_tform_time)
        
        # print("TFORM ", tform.shape, tform)
        # print("TFORM ", tform.shape)
        start_prepare_origimage = time.time()
        original_image = current_image['original_image'][None, ...].to(device)
        # NOTE: doesn't this call the getter two times with the related overhead
        _, image_height, image_width = current_image['original_image'].shape
        end_prepare_origimage = time.time()
        prepare_origimage_time = end_prepare_origimage - start_prepare_origimage
        prepare_origimage_times.append(prepare_origimage_time)
        
        end_prepare = time.time()
        prepare_time = end_prepare - start_prepare
        prepare_times.append(prepare_time)
                

        start_iteration = time.time()
        #breakpoint()
        with torch.no_grad():
            # ---- ENCODING ----
            start_encode = time.time()
            codedict = deca.encode(image)
            end_encode = time.time()
            encode_time = end_encode - start_encode
            total_encode_time += encode_time
            encoding_times.append(encode_time)
            
            # ---- DECODING ----
            start_decode = time.time()
             
            opdict = deca.decode_fast(codedict, original_image=original_image, tform=tform, use_detail=False, return_vis=False, full_res_landmarks2D=True)   

            end_decode = time.time()
            decode_time = end_decode - start_decode
            total_decode_time += decode_time
            decoding_times.append(decode_time)    
             
            # ---- LANDMARKS ----
            start_landmarks = time.time()

            landmarks3D = opdict['landmarks3d_world'][0].cpu().numpy()[:, :3]
            # print(f'\nImage {i}: - BEFORE SCALING eye_distance is {math.dist(landmarks3D[39], landmarks3D[42])}')
            scaling_factor = scalePoints(landmarks3D, 39, 42, 3)
            # print(f'\nImage {i}: - AFTER SCALING eye_distance is {math.dist(landmarks3D[39], landmarks3D[42])}')
            landmarks3D = np.ascontiguousarray(landmarks3D, dtype=np.float32)

            # if full_res_slow:
            #     landmarks2Dfullres = orig_visdict['landmarks2d_full_res'][0]
            # else:
            #     landmarks2Dfullres = opdict['landmarks2d_full_res'][0]
            
            landmarks2Dfullres = opdict['landmarks2d_full_res'][0]
            landmarks2Dfullres = np.ascontiguousarray(landmarks2Dfullres, dtype=np.float32)
            
            all_landmarks3D[i] = landmarks3D
            all_landmarks2D[i] = landmarks2Dfullres
            
            end_landmarks = time.time()
            landmarks_time = end_landmarks - start_landmarks
            total_landmark_time += landmarks_time
            landmark_times.append(landmarks_time)

            # print("BEFORE PNP - landmarks2Dfullres ",landmarks2Dfullres.shape,"landmarks3D",landmarks3D.shape)
            
            vertices = opdict['verts'][0].cpu().numpy()[:, :3]
            scalePoints(vertices, None, None, None, scaling_factor)

            if args.drawLandmarks2d:
                draw_points(i, original_image[0], landmarks2Dfullres, current_image)
            if args.saveObjWithLandmarks3d:
                save_obj_with_landmarks3d(i, landmarks3D, vertices)
            
            # ---- SOLVEPNP ----
            start_solvepnp = time.time()
            # Call solvePnP to estimate the pose
            #breakpoint()
            success, rotation_vector, translation_vector = cv2.solvePnP(
                landmarks3D, 
                #landmarks2D, 
                landmarks2Dfullres, 
                camera_matrix, 
                dist_coeffs)
                #flags=flags)
            if success:
                distance = np.linalg.norm(translation_vector)
                distances.append(distance)
            else:
                print(f'FAILED')
            # breakpoint()

            end_solvepnp = time.time()
            solvepnp_time = end_solvepnp - start_solvepnp
            total_solvepnp_time += solvepnp_time
            solvepnp_times.append(solvepnp_time)
            
            end_iteration = time.time()
            iteration_time = end_iteration - start_iteration
            iteration_times.append(iteration_time)
            # Verify : Reproject point from 3D to 2D
            if args.drawLandmarks2d:
                image_reprojection = original_image[0].clone()            
                landmarks2D_reproj, _ = cv2.projectPoints(landmarks3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
                draw_points(99, image_reprojection, np.squeeze(landmarks2D_reproj), current_image)
            # breakpoint()            

            # print("AFTER PNP - translation_vector ", translation_vector.shape, translation_vector)
            if args.saveObjPNP:
                save_obj_with_landmarks3d_rotated(i, landmarks3D, vertices, translation_vector, rotation_vector)
       
        # enable or disable to manage performances
        if args.saveLandmarks3dPly:
            save_landmarks3d_ply(i, landmarks3D)
        if args.saveLandmarksTxt:
            save_landmarks2d3d_txt(i, landmarks2Dfullres, landmarks3D)
        if args.saveDistances:
            save_distances(i, distance)
        # breakpoint()
       
    
# ---- outside loop, after all processing ended
    for i in range(len(testdata)):
        print(f"Image ",i," - Encode time -", encoding_times[i])
        print(f"Image ",i," - Decode time -", decoding_times[i])
        print(f"Image ",i," - Landmark time -", landmark_times[i])
        print(f"Image ",i," - Solvepnp time -", solvepnp_times[i])
        print(f"Image ",i," - Distance is -", distances[i])
    for i in range(len(testdata)):
        print(f"Image ",i," - Get image time is -", get_image_times[i])
        print(f"Image ",i," - Prepare image time is -", prepare_image_times[i])
        print(f"Image ",i," - Prepare tform time is -", prepare_tform_times[i])
        print(f"Image ",i," - Prepare orig image time is -", prepare_origimage_times[i])
        print(f"Image ",i," - PREPARE total time is -", prepare_times[i])
        print(f"Image ",i," - ITERATION TIME is -", iteration_times[i])
        print(f"Image ",i," - TOTAL TIME is -", prepare_times[i]+iteration_times[i])
        print(f"---------------------", )

    # print(f"Total encoding time for all images - {total_encode_time}")
    # print(f"Total decoding time for all images - {total_decode_time}")
    # print(f"Average encoding time per image - {total_encode_time / len(testdata)}")
    # print(f"Average decoding time per image - {total_decode_time / len(testdata)}")   
    # print(f"Total Landmark time for all images - {total_landmark_time}")
    # print(f"Total Solvepnp time for all images - {total_solvepnp_time}")
    
    if args.saveAllLandmarksAndDistances:
        save_allLandmarksAndDistances(all_landmarks3D, all_landmarks3D, distances)
    
def scalePoints(points, fromPointIndex, toPointIndex, desiredSize, desiredScalingFactor=None):
    if(fromPointIndex is not None and toPointIndex is not None):
         eyeDistance = math.dist(points[fromPointIndex], points[toPointIndex])
         scalingFactor = desiredSize/eyeDistance # 3 cm between real life corresponing points (eyes inner corners)
         points *= scalingFactor
    elif (desiredScalingFactor is not None):
        points *= desiredScalingFactor
        scalingFactor = desiredScalingFactor
    return scalingFactor

def convert_normalized_to_pixels(normalized_points, image_width, image_height, imageData):
    pixel_points = []
    bbox = imageData['bbox']
    left, top, right, bottom = bbox
    bbox_width = right - left
    bbox_height = bottom - top
    print("bbox is ",bbox)
    for x_normalized, y_normalized in normalized_points:
        # Convert from [-1, 1] range to [0, width] and [0, height] range respectively
        x_pixel = (x_normalized + 1.0) * image_width / 2.0
        y_pixel = (y_normalized + 1.0) * image_height / 2.0
        
        x_pixel = left + (x_pixel * bbox_width / image_width)
        y_pixel = top + (y_pixel * bbox_height / image_height)
        
        bbox_center_x = left + bbox_width / 2.0
        bbox_center_y = top + bbox_height / 2.0

        x_pixel = bbox_center_x + (x_pixel - bbox_center_x) * 1.25
        y_pixel = bbox_center_y + (y_pixel - bbox_center_y) * 1.25

        pixel_points.append((x_pixel, y_pixel))
    #breakpoint()
    return pixel_points

def draw_points(i, image_tensor, landmarks2D, imageData):
    image_numpy = (image_tensor.permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8)
    for point in landmarks2D:
         x, y = int(round(point[0])), int(round(point[1])) 
         cv2.circle(image_numpy, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
         
    bbox = imageData['bbox']
    left, top, right, bottom = bbox
    cv2.rectangle(image_numpy, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
    #breakpoint()

    cv2.imwrite(f"{global_args.savefolder}/{i}_with_landmarks.png", image_numpy)   
    
    #cv2.imshow('Landmarks', image_numpy)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

def save_landmarks3d_ply(i, landmarks3D):
    #breakpoint()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(landmarks3D)
    save_path = f"{global_args.savefolder}/{i}_kpt3d.ply"
    # Write the point cloud to a PLY file with the header
    o3d.io.write_point_cloud(save_path, pcd)


def save_obj_with_landmarks3d(i, landmarks3D, vertices):
    
    mesh_default = o3d.io.read_triangle_mesh(global_args.templateMeshPath)
    mesh_default.vertices = o3d.utility.Vector3dVector(vertices)
    o3d.io.write_triangle_mesh(f'{global_args.savefolder}/{i}_mesh_expression.ply', mesh_default)

    pc_landmarks3D_world = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(landmarks3D))
    o3d.io.write_point_cloud(f'{global_args.savefolder}/{i}_pc_landmarks3D_world.ply', pc_landmarks3D_world)
    
    pc_vertices = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vertices))
    o3d.io.write_point_cloud(f'{global_args.savefolder}/{i}_pc_vertices.ply', pc_vertices)

def save_obj_with_landmarks3d_rotated(i, landmarks3D, vertices, translation_vector, rotation_vector):

    # rotation_vector_deca = codedict['pose'].cpu().numpy()[0][:3]  # Extract the first three elements
    # rotation_matrix = cv2.Rodrigues(rotation_vector_deca)
    rotation_matrix = cv2.Rodrigues(rotation_vector)
    print("Rotation is ", rotation_matrix[0])
    landmarks_3D_rotated = np.dot(landmarks3D, rotation_matrix[0].T)
    vertices_rotated = np.dot(vertices, rotation_matrix[0].T)
    # original_mesh.vertices = np.dot(original_mesh.vertices, rotation_matrix[0].T)
    # breakpoint()
    
    # Translate original mesh in place where 3D landamrks are
    # original_mesh.vertices += translation_vector.T

    # original_mesh.export(f'{args.savefolder}/{i}_scaledModel.obj')

    mesh_default_rotated = o3d.io.read_triangle_mesh(global_args.templateMeshPath)
    mesh_default_rotated.vertices = o3d.utility.Vector3dVector(vertices_rotated)
    o3d.io.write_triangle_mesh(f'{global_args.savefolder}/{i}_mesh_expression_rotated.ply', mesh_default_rotated)

    pc_landmarks3D_world_rotated = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(landmarks_3D_rotated))
    o3d.io.write_point_cloud(f'{global_args.savefolder}/{i}_pc_landmarks3D_world_rotated.ply', pc_landmarks3D_world_rotated)
    
    pc_vertices_rotated = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vertices_rotated))
    o3d.io.write_point_cloud(f'{global_args.savefolder}/{i}_pc_vertices_rotated.ply', pc_vertices_rotated)
    
    # Compute the normals of the mesh (useful for rendering)
    mesh_default_rotated.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh_default_rotated])


def save_landmarks2d3d_txt(i, landmarks2Dfullres, landmarks3D):
    np.savetxt(f'{global_args.savefolder}/{i}_landmarks2d.txt', landmarks2Dfullres)
    np.savetxt(f'{global_args.savefolder}/{i}_landmarks3d.txt', landmarks3D)  
    
def save_distances(i, distance):
    np.savetxt(f'{global_args.savefolder}/{i}_distance.txt', distance)  
    
def save_allLandmarksAndDistances(all_landmarks3D, all_landmarks2D, distances):
    np.savez(f'{global_args.savefolder}/all_landmarks3D.npz', all_landmarks3D = all_landmarks3D)
    np.savez(f'{global_args.savefolder}/all_landmarks2D.npz', all_landmarks2D = all_landmarks2D)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA + SolvePnP')

    parser.add_argument('-i', '--inputpath', default='TestSamples/examples', type=str,
                        help='path to the test data, can be image folder, image path, image list, video')
    parser.add_argument('-s', '--savefolder', default='TestSamples/examples/results', type=str,
                        help='path to the output directory, where results(obj, txt files) will be stored.')
    parser.add_argument('--device', default='cuda', type=str,
                        help='set device, cpu for using cpu' )
    # process test images
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped' )
    parser.add_argument('--sample_step', default=10, type=int,
                        help='sample images from video data for every step' )
    parser.add_argument('--detector', default='fan', type=str,
                        help='detector for cropping face, check decalib/detectors.py for details' )
    # rendering option
    parser.add_argument('--rasterizer_type', default='standard', type=str,
                        help='rasterizer type: pytorch3d or standard' )
    # parser.add_argument('--render_orig', default=True, type=lambda x: x.lower() in ['true', '1'],
    #                     help='whether to render results in original image size, currently only works when rasterizer_type=standard')
    # save
    parser.add_argument('--useTex', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to use FLAME texture model to generate uv texture map, \
                            set it to True only if you downloaded texture model' )
    parser.add_argument('--extractTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to extract texture from input image as the uv texture map, set false if you want albeo map from FLAME mode' )
    # parser.add_argument('--saveVis', default=True, type=lambda x: x.lower() in ['true', '1'],
    #                     help='whether to save visualization of output' )
    parser.add_argument('--saveKpt', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save 2D and 3D keypoints' )
    # parser.add_argument('--saveDepth', default=False, type=lambda x: x.lower() in ['true', '1'],
    #                     help='whether to save depth image' )
    # parser.add_argument('--saveObj', default=False, type=lambda x: x.lower() in ['true', '1'],
    #                     help='whether to save outputs as .obj, detail mesh will end with _detail.obj. \
    #                         Note that saving objs could be slow' )
    parser.add_argument('--saveMat', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .mat' )
    parser.add_argument('--saveImages', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization output as seperate images' )
    
    # Additional Arguments not in original lib
    parser.add_argument('--templateMeshPath', default="data\head_template_centered.ply", type=str,
                        help='path to the ply template mesh for visualisation' )
    parser.add_argument('--calibrationData', default="Python Camera Calibration/calibration_data_webcam_side.npz", type=str,
                        help='calibration data with camera matrix and distortion coefficients as result of manual calibration' )
    parser.add_argument('--drawLandmarks2d', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='draw the full res landmarks2d on top of the original image' )
    parser.add_argument('--saveLandmarks3dPly', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='output a point cloud representation of the 3d landmarks' )
    parser.add_argument('--saveLandmarksTxt', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='save landmarks 2d and 3d as txt for each input image' )
    parser.add_argument('--saveDistances', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='save distance between face and camera for each input image' )
    parser.add_argument('--saveAllLandmarksAndDistances', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='save one big file for each of landmarks 2d, 3d, and distances, but with data for all input images' )
    parser.add_argument('--saveObjWithLandmarks3d', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='save for each img the obj morphed to the recognised verts, the verts point cloud, the landmarks3d point cloud' )
    parser.add_argument('--saveObjPNP', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='save the result of applying translation and rotation to the morphed obj' )
            
    main(parser.parse_args())