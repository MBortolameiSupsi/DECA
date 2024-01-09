import trimesh
import numpy as np

mesh = trimesh.load('./head_template2.obj')



print("at 3827 ", mesh.vertices[3827])
print(" at 3619 ", mesh.vertices[3619])

mesh.export('path_to_your_output_file.obj')

mesh2 = trimesh.load('./head_template2.obj')
print("at 3827 ", mesh2.vertices[3827])
print(" at 3619 ", mesh2.vertices[3619])