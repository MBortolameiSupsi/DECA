import trimesh
import numpy as np
import math

def main():

    mesh = trimesh.load('./head_template2.obj')
    meshOriginal = trimesh.load('./head_template.obj')


    pointA = mesh.vertices[3827]
    pointB = mesh.vertices[3619]
    
    print("at 3827 ", pointA)
    print(" at 3619 ", pointB)    
    
    foundIndexA = -1
    foundIndexB = -1
    
    for index,vertex in enumerate(meshOriginal.vertices):
        if(all(vertex == pointA)):
           foundIndexA = index
        if(all(vertex == pointB)):
           foundIndexB = index
        
    print("foundIndexA ", foundIndexA)
    print("foundIndexB", foundIndexB)
    # scalePoints(mesh.vertices, 3827, 3619, 3)

    # print("at 3827 ", mesh.vertices[3827])
    # print(" at 3619 ", mesh.vertices[3619])

def scalePoints(points, fromPointIndex, toPointIndex, desiredSize):
     eye_distance = math.dist(points[fromPointIndex], points[toPointIndex])
     scaling_factor = desiredSize/eye_distance # 3 cm between real life corresponing points (eyes inner corners)
     points *= scaling_factor

if __name__ == "__main__":
    main()