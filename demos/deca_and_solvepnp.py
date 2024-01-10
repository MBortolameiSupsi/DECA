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


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#full_res_slow = True
full_res_slow = False

#use_original = True
use_original = False

if use_original:
    from decalib.deca_original import DECA
else:
    from decalib.deca import DECA

from decalib.datasets import datasets 
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg
from decalib.utils.tensor_cropper import transform_points

def main(args):
    # if args.rasterizer_type != 'standard':
    #     args.render_orig = False
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

    print('camera_matrix', camera_matrix, '\n dist_coeffs', dist_coeffs)

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
    # results
    distances = []
    all_landmarks3D = np.empty((len(testdata), 68, 3))
    all_landmarks2D = np.empty((len(testdata), 68, 2))
    
    # Explicitly specify the flag (try different flags if necessary)
    # flags = cv2.SOLVEPNP_EPNP  # Example flag, adjust as needed

    print("FULL RES SLOW ", full_res_slow, "deca original ",use_original)
    for i in tqdm(range(len(testdata))):
        name = testdata[i]['imagename']
        image = testdata[i]['image'].to(device)[None,...]
        
        tform = testdata[i]['tform'][None, ...]
        tform = torch.inverse(tform).transpose(1,2).to(device)
        # print("TFORM ", tform.shape, tform)
        print("TFORM ", tform.shape)
        original_image = testdata[i]['original_image'][None, ...].to(device)

        _, image_height, image_width = testdata[i]['original_image'].shape
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
            
            if full_res_slow:
                opdict, orig_visdict = deca.decode(codedict, render_orig=True, original_image=original_image, tform=tform)               
            else:
                opdict = deca.decode_fast(codedict, original_image=original_image, tform=tform, use_detail=False, return_vis=False, full_res_landmarks2D=True)   
            #opdict, visdict = deca.decode(codedict) #tensor
            
            end_decode = time.time()
            decode_time = end_decode - start_decode
            total_decode_time += decode_time
            decoding_times.append(decode_time)    
             
            # ---- LANDMARKS ----
            start_landmarks = time.time()
            # landmarks3D = opdict['landmarks3d'][0].cpu().numpy()[:, :3]
            landmarks3D = opdict['landmarks3d_world'][0].cpu().numpy()[:, :3]
            #landmarks2D = opdict['landmarks2d'][0].cpu().numpy()
            
            print(f'\nImage {i}: - BEFORE SCALING eye_distance is {math.dist(landmarks3D[39], landmarks3D[42])}')
            scalePoints(landmarks3D, 39, 42, 3)
            landmarks3D = np.ascontiguousarray(landmarks3D, dtype=np.float32)
            print(f'\nImage {i}: - AFTER SCALING eye_distance is {math.dist(landmarks3D[39], landmarks3D[42])}')

            if full_res_slow:
                landmarks2Dfullres = orig_visdict['landmarks2d_full_res'][0]
            else:
                landmarks2Dfullres = opdict['landmarks2d_full_res'][0]
            
            landmarks2Dfullres = np.ascontiguousarray(landmarks2Dfullres, dtype=np.float32)
            
            #landmarks2D = convert_normalized_to_pixels(landmarks2D, 224, 224, testdata[i])
            #landmarks2D = np.ascontiguousarray(landmarks2D, dtype=np.float32)
            

            all_landmarks3D[i] = landmarks3D
            # all_landmarks2D[i] = landmarks2D         
            all_landmarks2D[i] = landmarks2Dfullres
            
            end_landmarks = time.time()
            landmarks_time = end_landmarks - start_landmarks
            total_landmark_time += landmarks_time
            landmark_times.append(landmarks_time)

            #print("opdict['landmarks3d'] ",opdict['landmarks3d'][0].cpu().numpy()[:, :3])
            # if full_res_slow:
            #     print("orig_visdict['landmarks3d'] ",orig_visdict['landmarks3d'][0].cpu().numpy()[:, :3])
            # print("BEFORE PNP - landmarks2Dfullres ",landmarks2Dfullres,"landmarks3D",landmarks3D.shape, landmarks3D)
            print("BEFORE PNP - landmarks2Dfullres ",landmarks2Dfullres.shape,"landmarks3D",landmarks3D.shape)
            
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
            
            # print("AFTER PNP - translation_vector ", translation_vector.shape, translation_vector)
            print("AFTER PNP - translation_vector ", translation_vector.shape)
                
        # enable or disable to manage performances
        if args.drawLandmarks2d:
            draw_points(i, original_image[0], landmarks2Dfullres, testdata[i], args)
            #draw_points(i, original_image[0], landmarks2D, testdata[i])
        if args.saveLandmarks3dPly:
            save_landmarks3d_ply(i, landmarks3D, args)
        if args.saveLandmarksTxt:
            save_landmarks2d3d_txt(i, landmarks2Dfullres, landmarks3D, args)
        if args.saveDistances:
            save_distances(i, distance, args)
        # show_obj_with_landmarks3d(i, landmarks3D_world, args)
        # breakpoint()
       
        show_obj_with_landmarks3d(i, landmarks3D, args, rotation_vector, translation_vector, codedict, opdict)
    
# ---- outside loop, after all processing ended
    for i in range(len(testdata)):
        print(f"Image ",i," - Encode time -", encoding_times[i])
        print(f"Image ",i," - Decode time -", decoding_times[i])
        print(f"Image ",i," - Landmark time -", landmark_times[i])
        print(f"Image ",i," - Solvepnp time -", solvepnp_times[i])
        print(f"Image ",i," - Distance is -", distances[i])

    print(f"Total encoding time for all images - {total_encode_time}")
    print(f"Total decoding time for all images - {total_decode_time}")
    print(f"Average encoding time per image - {total_encode_time / len(testdata)}")
    print(f"Average decoding time per image - {total_decode_time / len(testdata)}")   
    print(f"Total Landmark time for all images - {total_landmark_time}")
    print(f"Total Solvepnp time for all images - {total_solvepnp_time}")
    
    if args.saveAllLandmarksAndDistances:
        save_allLandmarksAndDistances(all_landmarks3D, all_landmarks3D, distances, args)
    
    # print('all_landmarks2D is ',all_landmarks2D)
    #breakpoint()
        # ---- ONLY TO PRINT ORIGINAL RESULTS ----
        # opdict, visdict = deca.decode(codedict) #tensor
        # tform = testdata[i]['tform'][None, ...]
        # tform = torch.inverse(tform).transpose(1,2).to(device)
        # original_image = testdata[i]['original_image'][None, ...].to(device)
        # _, orig_visdict = deca.decode(codedict, render_orig=True, original_image=original_image, tform=tform)    
        # orig_visdict['inputs'] = original_image 
           
#---- LEGACY ARGUMENTS FROM ORIGINAL DECA LIBRARY - use at own discretion ----
#         if args.saveDepth or args.saveKpt or args.saveObj or args.saveMat or args.saveImages:
#             os.makedirs(os.path.join(savefolder, name), exist_ok=True)
# ##        -- save results
#         if args.saveDepth:
#             depth_image = deca.render.render_depth(opdict['trans_verts']).repeat(1,3,1,1)
#             visdict['depth_images'] = depth_image
#             cv2.imwrite(os.path.join(savefolder, name, name + '_depth.jpg'), util.tensor2image(depth_image[0]))
        # if args.saveKpt:
        #      np.savetxt(os.path.join(savefolder, name, name + '_kpt2d.txt'), opdict['landmarks2d'][0].cpu().numpy())
        #      np.savetxt(os.path.join(savefolder, name, name + '_kpt3d.txt'), opdict['landmarks3d'][0].cpu().numpy())
#         if args.saveObj:
#             deca.save_obj(os.path.join(savefolder, name, name + '.obj'), opdict)
#         if args.saveMat:
#             opdict = util.dict_tensor2npy(opdict)
#             savemat(os.path.join(savefolder, name, name + '.mat'), opdict)
#         if args.saveVis:
#             cv2.imwrite(os.path.join(savefolder, name + '_vis.jpg'), deca.visualize(visdict))
#             if args.render_orig:
#                 print("saving vis")
#                 cv2.imwrite(os.path.join(savefolder, name + '_vis_original_size.jpg'), deca.visualize(orig_visdict))
#         if args.saveImages:
#             for vis_name in ['inputs', 'rendered_images', 'albedo_images', 'shape_images', 'shape_detail_images', 'landmarks2d']:
#                 if vis_name not in visdict.keys():
#                     continue
#                 image = util.tensor2image(visdict[vis_name][0])
#                 print("saving images 1")
#                 cv2.imwrite(os.path.join(savefolder, name, name + '_' + vis_name +'.jpg'), util.tensor2image(visdict[vis_name][0]))
#                 if args.render_orig:
#                     print("saving images 2")
#                     image = util.tensor2image(orig_visdict[vis_name][0])
#                     cv2.imwrite(os.path.join(savefolder, name, 'orig_' + name + '_' + vis_name +'.jpg'), util.tensor2image(orig_visdict[vis_name][0]))
    # print(f'-- please check the results in {savefolder}')

def scalePoints(points, fromPointIndex, toPointIndex, desiredSize):
     eye_distance = math.dist(points[fromPointIndex], points[toPointIndex])
     scaling_factor = desiredSize/eye_distance # 3 cm between real life corresponing points (eyes inner corners)
     points *= scaling_factor
     return scaling_factor

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

def draw_points(i, image_tensor, landmarks2D, imageData, args):
    image_numpy = (image_tensor.permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8)
    for point in landmarks2D:
         x, y = int(round(point[0])), int(round(point[1])) 
         cv2.circle(image_numpy, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
         
    bbox = imageData['bbox']
    left, top, right, bottom = bbox
    cv2.rectangle(image_numpy, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
    #breakpoint()

    cv2.imwrite(f"{args.savefolder}/result_{i}_with_landmarks.png", image_numpy)   
    
    #cv2.imshow('Landmarks', image_numpy)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

def save_landmarks3d_ply(i, landmarks3D, args):
    #breakpoint()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(landmarks3D)
    save_path = f"{args.savefolder}/{i}_kpt3d.ply"
    # Write the point cloud to a PLY file with the header
    o3d.io.write_point_cloud(save_path, pcd)
    

def show_obj_with_landmarks3d(i, landmarks3D, args, rotation_vector, translation_vector, codedict, opdict):
    # Load the .obj file
    original_mesh = trimesh.load('data/head_template_centered.obj')
    
    # rotation_vector_deca = codedict['pose'].cpu().numpy()[0][:3]  # Extract the first three elements
    # rotation_matrix = cv2.Rodrigues(rotation_vector_deca)
    rotation_matrix = cv2.Rodrigues(rotation_vector)
    print("Rotation is ", rotation_matrix[0])
    original_mesh.vertices = np.dot(original_mesh.vertices, rotation_matrix[0].T)
    # breakpoint()
    original_mesh.vertices += translation_vector.T
    
    print(f'\nObj resizing: - BEFORE SCALING eye_distance is {math.dist(original_mesh.vertices[3858], original_mesh.vertices[3649])}')
    # Resize mesh to match 3cm distance between eye corners
    # scalePoints(original_mesh.vertices, 3827, 3619, 3)
    scalePoints(original_mesh.vertices, 3858, 3649, 3)
    print(f'\nObj resizing: - AFTER SCALING eye_distance is {math.dist(original_mesh.vertices[3858], original_mesh.vertices[3649])}')
    # Create a list to hold all meshes (original mesh + spheres)
    original_mesh.export(f'{args.savefolder}/{i}scaledModel.obj')

    all_meshes = [original_mesh]
    empty_obj = trimesh.PointCloud(vertices=[])
    landmarks3D_spheres = [empty_obj]
    landmarks3D_original_spheres = [empty_obj]
    verts_spheres = [empty_obj]
    trans_verts_spheres = [empty_obj]
    # all_spheres = [empty_obj]

    # Sphere parameters (you can adjust the radius and subdivisions)
    sphere_radius = 0.05
    sphere_subdivisions = 2
    red = (255,0,0)
    green = (0,255,0)
    blue = (0,0,255)
    purple = (128,0,128)

    # Create a sphere for each landmark and add it to the list of meshes
    # landmarks3D are WORLD
    for landmark in landmarks3D:
        sphere = create_sphere_at(center=landmark, radius=sphere_radius, subdivisions=sphere_subdivisions, color=red)
        all_meshes.append(sphere)
        landmarks3D_spheres.append(sphere)
        
    landmarks3D_original = opdict['landmarks3d'][0].cpu().numpy()[:, :3]
    scalePoints(landmarks3D_original, 39, 42, 3)
    for landmark in landmarks3D_original:
        sphere = create_sphere_at(center=landmark, radius=sphere_radius, subdivisions=sphere_subdivisions, color=green)
        landmarks3D_original_spheres.append(sphere)

    verts = opdict['verts'][0].cpu().numpy()[:, :3]
    scalePoints(verts, 3827, 3619, 3)
    for vert in verts:
        sphere = create_sphere_at(center=vert, radius=sphere_radius, subdivisions=sphere_subdivisions, color=blue)
        verts_spheres.append(sphere)    
    
    trans_verts = opdict['trans_verts'][0].cpu().numpy()[:, :3]
    scalePoints(trans_verts, 3827, 3619, 3)
    for vert in trans_verts:
        sphere = create_sphere_at(center=vert, radius=sphere_radius, subdivisions=sphere_subdivisions, color=purple)
        trans_verts_spheres.append(sphere)    
    # for vertex in original_mesh.vertices:
    #     sphere = create_sphere_at(center=vertex, radius=sphere_radius, subdivisions=sphere_subdivisions)
    #     all_spheres.append(sphere)

    # Combine all meshes into a single mesh
    all_meshes = trimesh.util.concatenate(all_meshes)
    landmarks3d_mesh = trimesh.util.concatenate(landmarks3D_spheres[1:])  # Skip the first empty PointCloud    # Export the combined mesh to a new OBJ file
    landmarks3D_original_mesh = trimesh.util.concatenate(landmarks3D_original_spheres[1:])  # Skip the first empty PointCloud    # Export the combined mesh to a new OBJ file
    verts_mesh = trimesh.util.concatenate(verts_spheres[1:])  # Skip the first empty PointCloud    # Export the combined mesh to a new OBJ file
    trans_verts_mesh = trimesh.util.concatenate(trans_verts_spheres[1:])  # Skip the first empty PointCloud    # Export the combined mesh to a new OBJ file
    # all_spheres_mesh = trimesh.util.concatenate(all_spheres[1:])  # Skip the first empty PointCloud    # Export the combined mesh to a new OBJ file
    
    all_meshes.export(f'{args.savefolder}/{i}all_meshes.obj')
    landmarks3d_mesh.export(f'{args.savefolder}/{i}landmarks3d_mesh.obj')
    landmarks3D_original_mesh.export(f'{args.savefolder}/{i}landmarks3D_original_mesh.obj')
    verts_mesh.export(f'{args.savefolder}/{i}verts_mesh.obj')
    trans_verts_mesh.export(f'{args.savefolder}/{i}trans_verts_mesh.obj')
    # all_spheres_mesh.export(f'{args.savefolder}/{i}all_spheres_modelWithLandmarks3D.obj')

# Function to plot a sphere at each landmark point
def create_sphere_at(center, radius, subdivisions=3, color=(255,0,0) ):
    # Create a sphere mesh
    sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    # Translate the sphere to the landmark position
    sphere.apply_translation(center)
    
    # Apply the color to each vertex of the sphere
    # Vertex colors need to be in the shape of (num_vertices, 4)
    sphere.visual.vertex_colors = np.array([color] * len(sphere.vertices), dtype=np.uint8)
                                           
    return sphere

def save_landmarks2d3d_txt(i, landmarks2Dfullres, landmarks3D, args):
    np.savetxt(f'{args.savefolder}/{i}_landmarks2d.txt', landmarks2Dfullres)
    np.savetxt(f'{args.savefolder}/{i}_landmarks3d.txt', landmarks3D)  
    
def save_distances(i, distance, args):
    np.savetxt(f'{args.savefolder}/{i}_distance.txt', distance)  
    
def save_allLandmarksAndDistances(all_landmarks3D, all_landmarks2D, distances, args):
    suffix = 'slow' if full_res_slow else 'fast'
    np.savez(f'{args.savefolder}/all_landmarks3D'+ suffix+'.npz', all_landmarks3D = all_landmarks3D)
    np.savez(f'{args.savefolder}/all_landmarks2D'+ suffix+'.npz', all_landmarks2D = all_landmarks2D)

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
            
    main(parser.parse_args())