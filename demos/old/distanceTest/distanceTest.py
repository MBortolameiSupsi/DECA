import numpy as np 


tvecs = [ [0,0,1000],[0,0,1000],[0,0,1000],[0,0,1000] ]

for t in tvecs:
    print(f"distance is {np.linalg.norm(t)}")