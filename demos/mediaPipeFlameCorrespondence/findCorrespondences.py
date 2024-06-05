import open3d as o3d
import numpy as np
from scipy.spatial import KDTree
import os

def load_ply(filepath):
    pc = o3d.io.read_point_cloud(filepath)
    return np.asarray(pc.points)

script_path = os.path.dirname(__file__)

# Load the point clouds
mediapipe_mesh = load_ply(os.path.join(script_path, "mp_scaled.ply"))
deca_mesh = load_ply(os.path.join(script_path, "deca_scaled.ply"))

# Build a KDTree for the deca set of vertices
tree = KDTree(deca_mesh)

# Find the nearest neighbor in deca_mesh for each vertex in mediapipe_mesh
distances, indices = tree.query(mediapipe_mesh)
# Writing indices to the file
with open(os.path.join(script_path, "correspondences.txt"), 'w') as file:
    for i, index in enumerate(indices):
        file.write(f"{i}, {index}\n")

# Create line set for visualization
lines = [[i, i + len(mediapipe_mesh)] for i in range(len(mediapipe_mesh))]  # Line between corresponding points
line_colors = [[0, 0, 1] for i in range(len(lines))]  # Blue lines

# Combine both sets of vertices for a unified point cloud
all_points = np.vstack((mediapipe_mesh, deca_mesh[indices]))

# Colors for each point set
mediapipe_colors = [[0, 1, 0] for _ in range(len(mediapipe_mesh))]  # Green for mediapipe
deca_colors = [[1, 0, 0] for _ in range(len(deca_mesh[indices]))]  # Red for deca
all_colors = mediapipe_colors + deca_colors

# Create a point cloud object for visualization
pcd_all_points = o3d.geometry.PointCloud()
pcd_all_points.points = o3d.utility.Vector3dVector(all_points)
pcd_all_points.colors = o3d.utility.Vector3dVector(all_colors)

# Create a line set object
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(all_points)
line_set.lines = o3d.utility.Vector2iVector(lines)
line_set.colors = o3d.utility.Vector3dVector(line_colors)

# Visualize
# o3d.visualization.draw_geometries([pcd_all_points, line_set])

# Assuming `line_set` is your LineSet object
success = o3d.io.write_line_set(os.path.join(script_path, "correspondences.ply"), line_set, print_progress=True)
