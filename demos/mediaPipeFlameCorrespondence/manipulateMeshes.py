from re import S
import numpy as np
import cv2
import open3d as o3d
import os
import math

def main():
    script_path = os.path.dirname(__file__)

    # Load the point cloud
    mp_cloud = o3d.io.read_point_cloud(os.path.join(script_path, "mediapipe_landmarks.ply"))
    mp_points = np.asarray(mp_cloud.points)
    # Scale it to 3cm between eyes
    mp_points, _ = scalePoints(mp_points,173,398, 3)
    # Rotate around x axis by 180
    theta = math.radians(180)
    rot_matrix_x = rotation_matrix('x', theta)
    mp_points = mp_points.dot(rot_matrix_x.T)
    # Translate left eye index to zero 
    translation_vector = -mp_points[173]
    mp_points += translation_vector
    # Save
    mp_cloud.points = o3d.utility.Vector3dVector(mp_points)
    o3d.io.write_point_cloud(os.path.join(script_path, "mp_scaled.ply"), mp_cloud)
 
    # Load the 3d mesh with shape based on same image source
    deca_mesh = o3d.io.read_point_cloud(os.path.join(script_path, "deca_mesh.ply"))
    deca_vertices = np.asarray(deca_mesh.points)
    # Scale the vertices to 3cm between eyes
    deca_vertices, _ = scalePoints(deca_vertices, 3619, 3827, 3)
    # Translate left eye index to zero 
    translation_vector = -deca_vertices[3619]
    deca_vertices += translation_vector

    deca_mesh.points = o3d.utility.Vector3dVector(deca_vertices)
    o3d.io.write_point_cloud(os.path.join(script_path, "deca_scaled.ply"), deca_mesh)
    
    
    
    
    


def scalePoints(
    input_points,
    fromPointIndex=None,
    toPointIndex=None,
    desiredDistance=None,
    desiredScalingFactor=None,
):
   
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

def rotation_matrix(axis, theta):
    if axis == 'x':
        return np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])
    elif axis == 'y':
        return np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
    elif axis == 'z':
        return np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
    

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="DECA + SolvePnP")

    # # Additional Arguments not in original lib
    # parser.add_argument(
    #     "--config",
    #     default=os.path.join(script_dir,"./config_folder.yaml"),
    #     type=str,
    #     help="path to the config file",
    # )

    main()