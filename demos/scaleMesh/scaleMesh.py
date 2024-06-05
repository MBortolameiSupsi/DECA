import argparse
import os
import open3d as o3d
import trimesh
import math
import numpy as np

def main(args):
    # Load the mesh
    
    folder, file_name, file_format = check_file_format(args.meshPath)
    
    mesh, points = load_mesh(args.meshPath, file_format)        
    
    scaled_points, _ = scale_points(points, args.fromPointIndex, args.toPointIndex, args.desiredDistance)
    
    scaled_mesh = save_mesh(mesh, scaled_points, folder, file_name, file_format)
    
    
    
def save_mesh(mesh, scaled_points, folder, file_name, file_format):
    scaled_mesh_name = f"{file_name}_scaled.{file_format}"
    full_path = os.path.join(folder, scaled_mesh_name)
    if file_format == "obj":
        mesh.vertices = scaled_points
        mesh.export(full_path, file_type="obj")
    elif file_format == "ply":
        mesh.vertices = scaled_points
        o3d.io.write_triangle_mesh(full_path, mesh)
    
    return mesh
        
def load_mesh(mesh_path, file_format):
    if file_format == "obj":
        mesh = load_obj(mesh_path)
    elif file_format == "ply":
        mesh = load_ply(mesh_path)
        
    points = mesh.vertices
    return mesh, points
def load_ply(file_path):
    mesh = o3d.io.read_triangle_mesh(file_path)

    # Check if the mesh contains vertex normals (important for rendering and lighting)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    return mesh
def load_obj(file_path):
    mesh = trimesh.load(file_path)
    return mesh

def check_file_format(file_path):
    # Extract the folder and the full filename
    folder, full_filename = os.path.split(file_path)
    
    # Extract the filename without extension and the extension
    filename, extension = os.path.splitext(full_filename)
    
    # Normalize the extension to a standard lowercase format
    extension = extension.lower()
    
    # Determine the file format based on the extension
    if extension == '.obj':
        file_format = "obj"
    elif extension == '.ply':
        file_format = "ply"
    else:
        file_format = "Unknown format"
        
    return folder, filename, file_format
    
def scale_points(
    points,
    fromPointIndex=None,
    toPointIndex=None,
    desiredDistance=None,
    # desiredScalingFactor=None,
):
    mesh_points = np.asarray(points)
    if fromPointIndex != None and toPointIndex != None:
        currentEyeDistance = math.dist(mesh_points[fromPointIndex], mesh_points[toPointIndex])
        scalingFactor = desiredDistance / currentEyeDistance
        # myPrint(f"SCALING - initial distance was {currentEyeDistance} - desired is {desiredDistance} : scalingFactor {scalingFactor}")
        mesh_points *= scalingFactor
    # elif desiredScalingFactor != None:
    #     points *= desiredScalingFactor
    #     scalingFactor = desiredScalingFactor
    return mesh_points, scalingFactor



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scale a mesh")

    parser.add_argument(
        "--fromPointIndex",
        type=int,
        help="Starting point index from which to scale",
        # default= 362
        default= 1336
    )

    parser.add_argument(
        "--toPointIndex",
        type=int,
        help="Ending point index from which to scale",
        # default= 133
        default= 2288
    )
    parser.add_argument(
        "--desiredDistance",
        type=int,
        help="Desired distance between the points",
        default=3
    )    
    parser.add_argument(
        "--meshPath",
        type=str,
        help="Path to the mesh to be scaled",
        # default="C:/Users/massimo.bortolamei/Documents/head-tracking/data/mapping/mediapipeBaseModel.obj"
        default="C:/Users/massimo.bortolamei/Documents/head-tracking/data/mapping/map_aligned.obj"
    )
    # parser.add_argument(
    #     "--desiredScalingFactor",
    #     type=int,
    #     help="Desired Scaling factor for all points",
    # )
    main(parser.parse_args())