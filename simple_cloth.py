import numpy as np
from pathlib import Path
from pbrt_renderer import create_folder, to_real_array
import taichi as ti
from tqdm import tqdm
ti.init(arch=ti.cpu, default_fp=ti.f64)
from deformable import DeformableSimulator
#import os
#os.environ['TAICHI_MAX_NUM_SNODES'] = '10240000000'

X = 10
Y = 30
dx = 0.02
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
simulator.Initialize(init_vertices, elements, 1e3, 2e6, 0.3)
print("Initialization finished.")

# for j in range(X + 1):
#     simulator.dirichlet_boundary_condition[2 * j] = simulator.position[2 * j]
#     simulator.dirichlet_boundary_condition[2 * j + 1] = simulator.position[2 * j + 1]

# @ti.kernel
# def set_gravity():
for index in range(simulator.vertices_num):
    simulator.external_acceleration[index] = ti.Vector([0.0, 0.0, -9.8])

element_np = simulator.undeformed.elements.to_numpy()
folder = Path("./") / "simple_cloth"
create_folder(folder, exist_ok=True)
np.save(folder / "elements.npy", element_np)
position_0 = simulator.position.to_numpy()
np.save(folder / "0000.npy", position_0)

# @ti.kernel
# def set_force(a: ti.f64):
#     for i, j in ti.ndrange(Y + 1, X + 1):
#         mu = 0.5
#         simulator.external_acceleration[(i*(X+1) + j)*2] = ti.Vector([0.0, 0.0, a * ti.exp(-mu * ti.abs(i - Y))])
#         simulator.external_acceleration[(i*(X+1) + j)*2+1] = ti.Vector([0.0, 0.0, a * ti.exp(-mu * ti.abs(i - Y))])

#set_gravity()
print(simulator.external_acceleration.to_numpy())
test_pos = ti.field(dtype=ti.f64,shape=(vertices_num,3))
test_pos_1 = ti.field(dtype=ti.f64,shape=(vertices_num,3))
test_pos_zero = ti.field(dtype=ti.f64,shape=(vertices_num,3))


for f in tqdm(range(300)):
    position_np = simulator.position.to_numpy()
    print("Current step is {} and current position is {}".format(f,position_np))
    # set_force(ti.cos(f * 0.1) * 5)
    simulator.Forward(0.01)
    position_np = simulator.position.to_numpy()
    np.save(folder / "{:04d}.npy".format(f + 1), position_np)