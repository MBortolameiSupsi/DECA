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

import numpy as np
import torch
# import dlib

class FAN(object):
    def __init__(self):
        import face_alignment
        self.model = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device='cuda')

    def run(self, image):
        
        '''
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box list
        '''
        out = self.model.get_landmarks(image)
        if out is None:
            return [0], 'kpt68'
        else:
            kpt = out[0].squeeze()
            left = np.min(kpt[:,0]); right = np.max(kpt[:,0]); 
            top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])
            bbox = [left,top, right, bottom]
            return bbox, 'kpt68'
        
class MEDIAPIPE(object):
    def __init__(self):
        import mediapipe as mp
        import cv2
        mp_face_mesh = mp.solutions.face_mesh
        self.cv2 = cv2
        self.model = mp_face_mesh.FaceMesh()

    def run(self, image):
        out = self.model.process(self.cv2.cvtColor(image, self.cv2.COLOR_BGR2RGB))
        # out = self.model.get_landmarks(image)
        if out.multi_face_landmarks is None:
            return [0], 'kpt68'
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
        

class MTCNN(object):
    def __init__(self, device = 'cpu'):
        '''
        https://github.com/timesler/facenet-pytorch/blob/master/examples/infer.ipynb
        '''
        from facenet_pytorch import MTCNN as mtcnn
        self.device = device
        self.model = mtcnn(keep_all=True)
    def run(self, input):
        '''
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box
        '''
        out = self.model.detect(input[None,...])
        if out[0][0] is None:
            return [0]
        else:
            bbox = out[0][0].squeeze()
            return bbox, 'bbox'



