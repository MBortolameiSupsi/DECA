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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
full_res_slow = True
full_res_slow = False

use_original = True
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
    data = np.load(r'C:\Users\massimo.bortolamei\Documents\Python Camera Calibration\calibration_data_webcam_side.npz')
    #data = np.load(r'C:\Users\massimo.bortolamei\Documents\Python Camera Calibration\calibration_data_iphone_back_2.npz')
    camera_matrix = data['camera_matrix']
    dist_coeffs = data['dist_coeffs']
    # Ensure that camera_matrix and dist_coeffs are also float32 and have the correct shapes
    camera_matrix = np.asarray(camera_matrix, dtype=np.float32)
    dist_coeffs = np.asarray(dist_coeffs, dtype=np.float32)

    print('camera_matrix', camera_matrix, '\n dist_coeffs', dist_coeffs)
    # run DECA
    deca_cfg.model.use_tex = args.useTex
    deca_cfg.rasterizer_type = args.rasterizer_type
    deca_cfg.model.extract_tex = args.extractTex
    deca = DECA(config = deca_cfg, device=device)
    total_encode_time = 0
    total_decode_time = 0
    total_landmark_time = 0
    total_solvepnp_time = 0
    encoding_times = []
    decoding_times = []
    landmark_times = []
    solvepnp_times = []
    distances = []
    all_landmarks3D = np.empty((len(testdata), 68, 3))
    all_landmarks2D = np.empty((len(testdata), 68, 2))
    
    # Explicitly specify the flag (try different flags if necessary)
    #flags = cv2.SOLVEPNP_EPNP  # Example flag, adjust as needed
    # for i in range(len(testdata)):
    print("FULL RES SLOW ", full_res_slow, "deca original ",use_original)
    for i in tqdm(range(len(testdata))):
        name = testdata[i]['imagename']
        image = testdata[i]['image'].to(device)[None,...]
        
        tform = testdata[i]['tform'][None, ...]
        tform = torch.inverse(tform).transpose(1,2).to(device)
        print("TFORM ", tform.shape, tform)
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
            landmarks3D = opdict['landmarks3d'][0].cpu().numpy()[:, :3]
            #landmarks2D = opdict['landmarks2d'][0].cpu().numpy()
            
            eye_distance = math.dist(landmarks3D[39], landmarks3D[42])
            scaling_factor = 3/eye_distance # in cm
            landmarks3D = np.ascontiguousarray(landmarks3D, dtype=np.float32) * scaling_factor
            eye_distance2 = math.dist(landmarks3D[39], landmarks3D[42])
            print(f'\nImage {i}: eye distance {eye_distance} , scaling factor {scaling_factor} - AFTER SCALING eye_distance2 {eye_distance2}')

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
            #breakpoint()
            print("BEFORE PNP - landmarks2Dfullres ",landmarks2Dfullres,"landmarks3D",landmarks3D.shape, landmarks3D)
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
            
            end_solvepnp = time.time()
            solvepnp_time = end_solvepnp - start_solvepnp
            total_solvepnp_time += solvepnp_time
            solvepnp_times.append(solvepnp_time)
            
            print("AFTER PNP - translation_vector ", translation_vector.shape, translation_vector)
                
              
            draw_points(i, original_image[0], landmarks2Dfullres, testdata[i])
            save_landmarks3d_ply(i, landmarks3D)
            #draw_points(i, original_image[0], landmarks2D, testdata[i])
        ## ONLY TO PRINT ORIGINAL RESULTS ##
        # opdict, visdict = deca.decode(codedict) #tensor
        # tform = testdata[i]['tform'][None, ...]
        # tform = torch.inverse(tform).transpose(1,2).to(device)
        # original_image = testdata[i]['original_image'][None, ...].to(device)
        # _, orig_visdict = deca.decode(codedict, render_orig=True, original_image=original_image, tform=tform)    
        # orig_visdict['inputs'] = original_image 
 
#         if args.saveDepth or args.saveKpt or args.saveObj or args.saveMat or args.saveImages:
#             os.makedirs(os.path.join(savefolder, name), exist_ok=True)
# ##        -- save results
#         if args.saveDepth:
#             depth_image = deca.render.render_depth(opdict['trans_verts']).repeat(1,3,1,1)
#             visdict['depth_images'] = depth_image
#             cv2.imwrite(os.path.join(savefolder, name, name + '_depth.jpg'), util.tensor2image(depth_image[0]))
#         if args.saveKpt:
#             np.savetxt(os.path.join(savefolder, name, name + '_kpt2d.txt'), opdict['landmarks2d'][0].cpu().numpy())
#             np.savetxt(os.path.join(savefolder, name, name + '_kpt3d.txt'), opdict['landmarks3d'][0].cpu().numpy())
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
    
    suffix = 'slow' if full_res_slow else 'fast'
    np.savez('./demos/all_landmarks3D'+ suffix+'.npz', all_landmarks3D = all_landmarks3D)
    np.savez('./demos/all_landmarks2D'+ suffix+'.npz', all_landmarks2D = all_landmarks2D)
    # print('all_landmarks2D is ',all_landmarks2D)
    #breakpoint()

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

    cv2.imwrite(f"./TestSamples/result_{i}_with_landmarks.png", image_numpy)   
    
    #cv2.imshow('Landmarks', image_numpy)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

def draw_landmarks(i, landmarks2Doriginal):
    landmarks2Doriginal = landmarks2Doriginal.transpose(1, 2, 0)
    landmarks2Doriginal = (landmarks2Doriginal * 255).astype(np.uint8)
    cv2.imwrite(f"./TestSamples/original_result_{i}_with_landmarks.png", landmarks2Doriginal)     

def save_landmarks3d_ply(i, landmarks3D):
    #breakpoint()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(landmarks3D)
    save_path = f"./TestSamples/{i}_kpt3d.ply"
    # Write the point cloud to a PLY file with the header
    o3d.io.write_point_cloud(save_path, pcd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')

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
    parser.add_argument('--render_orig', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to render results in original image size, currently only works when rasterizer_type=standard')
    # save
    parser.add_argument('--useTex', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to use FLAME texture model to generate uv texture map, \
                            set it to True only if you downloaded texture model' )
    parser.add_argument('--extractTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to extract texture from input image as the uv texture map, set false if you want albeo map from FLAME mode' )
    parser.add_argument('--saveVis', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization of output' )
    parser.add_argument('--saveKpt', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save 2D and 3D keypoints' )
    parser.add_argument('--saveDepth', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save depth image' )
    parser.add_argument('--saveObj', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .obj, detail mesh will end with _detail.obj. \
                            Note that saving objs could be slow' )
    parser.add_argument('--saveMat', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .mat' )
    parser.add_argument('--saveImages', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization output as seperate images' )
    main(parser.parse_args())