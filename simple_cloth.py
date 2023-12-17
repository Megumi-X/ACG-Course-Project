import numpy as np
from pathlib import Path
from pbrt_renderer import create_folder, to_real_array
import taichi as ti
from tqdm import tqdm
ti.init(arch=ti.cuda)
from deformable import DeformableSimulator
#import os
#os.environ['TAICHI_MAX_NUM_SNODES'] = '10240000000'

X = 1
Y = 1
dx = 1.
vertices_num = (X + 1) * (Y + 1) * 2
elements_num = X * Y * 6
init_vertices = ti.Vector.field(n=3,dtype=ti.f64,shape=vertices_num)
for i in range(Y + 1):
    for j in range(X + 1):
        init_vertices[(i*(X+1) + j)*2] = ti.Vector([j * dx, i * dx, 0.0])
        init_vertices[(i*(X+1) + j)*2 + 1] = ti.Vector([j * dx, i * dx, dx])
elements = ti.Vector.field(n=4,dtype=ti.i32,shape=elements_num)
for i in range(Y):
    for j in range(X):
        a = (i * (X + 1) + j) * 2
        b = a + 1
        c = a + 2
        d = a + 3
        e = a + 2 * (X + 1)
        f = e + 1
        g = e + 2
        h = e + 3
        elements[(i*X + j)*6] = ti.Vector([a,c,d,h])
        elements[(i*X + j)*6 + 1] = ti.Vector([a,b,d,h])
        elements[(i*X + j)*6 + 2] = ti.Vector([a,b,f,h])
        elements[(i*X + j)*6 + 3] = ti.Vector([a,e,f,h])
        elements[(i*X + j)*6 + 4] = ti.Vector([a,e,g,h])
        elements[(i*X + j)*6 + 5] = ti.Vector([a,c,g,h])

print("Initializing...")
simulator = DeformableSimulator(vertices_num, elements_num)
simulator.Initialize(init_vertices, elements, 1e3, 1e2, 0.3)
print("Initialization finished.")

for j in range(X + 1):
    simulator.dirichlet_boundary_condition[2 * j] = ti.Vector([-1., -1., -1.])
    simulator.dirichlet_boundary_condition[2 * j + 1] = ti.Vector([3., 3., 3.])

for index in range(simulator.vertices_num):
    simulator.external_acceleration[index] = ti.Vector([0.0, 0.0, -0.0]) # no g

element_np = simulator.undeformed.elements.to_numpy()
folder = Path("./") / "simple_cloth"
create_folder(folder, exist_ok=True)
np.save(folder / "elements.npy", element_np)
position_0 = simulator.position.to_numpy()
np.save(folder / "0000.npy", position_0)

for i in tqdm(range(2000)):
    position_np = simulator.position.to_numpy()
    print("Current step is {} and current position is {}".format(i,position_np))
    simulator.Forward(0.01)
    position_np = simulator.position.to_numpy()
    np.save(folder / "{:04d}.npy".format(i + 1), position_np)