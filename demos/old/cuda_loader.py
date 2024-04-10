import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp, resize, rescale

def get_image():
    # Read the image using cv2
    # image = cv2.imread("C:\Users\massimo.bortolamei\Documents\DECA\TestSamples\examples_webcam\img (3).jpg")
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = np.array(imread("C:\Users\massimo.bortolamei\Documents\DECA\TestSamples\examples_webcam\img (3).jpg"))

    # Your existing conditions remain the same
    if len(image.shape) == 2:
        image = image[:, :, None].repeat(1, 1, 3)
    if len(image.shape) == 3 and image.shape[2] > 3:
        image = image[:, :, :3]

    h, w, _ = image.shape

    # Convert the image to float and normalize it if not already done

    # Rest of the code for transformations and other processing steps
    # ...
    src_pts = np.array([[0, 0], [0, h-1], [w-1, 0]])
    DST_PTS = np.array([[0,0], [224 - 1], [224 - 1, 0]])

    tform = estimate_transform('similarity', src_pts, DST_PTS)

    image = image.astype(np.float32) / 255.0
    dst_image = warp(image, tform.inverse, output_shape=(224, 224))
    dst_image = dst_image.transpose(2, 0, 1)

    return {
        'image': torch.tensor(dst_image).float(),
        'tform': torch.tensor(tform.params).float(),
        'original_image': torch.tensor(image.transpose(2, 0, 1)).float(),  # Transpose to CxHxW format
    }

def main():
    data = get_image()
    image = data['image'].to('cuda')[None,...]
    