import numpy as np
from pathlib import Path
from pbrt_renderer import create_folder, to_real_array
import taichi as ti
from tqdm import tqdm
ti.init(arch=ti.cuda)
from deformable import DeformableSimulator
simulator = DeformableSimulator(4,1)
init_vertices = ti.Vector.field(n=3,dtype=ti.f32,shape=4)
#init_vertices_vector = ti.Vector([[0.0,1.,0.],[0.0,0.0,0.],[1.,0.0,0.],[1.,1.,0.],[0.0,1.,1.],[0.0,0.0,1.],[1.,0.0,1.],[1.,1.,1.],[2.,2.,3.],[2.,3.,3.]])

# for idx in range(10):
#     for j in range(3):
#         init_vertices[idx,j] = init_vertices_vector[idx,j]
init_vertices[0] = ti.Vector([0.0, 0.0, 0.0])
init_vertices[1] = ti.Vector([1.0, 0.0, 0.0])
init_vertices[2] = ti.Vector([0.0, 1.0, 0.0])
init_vertices[3] = ti.Vector([0.0, 0.0, 1.0])


init_elements = ti.Vector.field(n=4,dtype=ti.i32,shape=1)
init_elements[0] = ti.Vector([0,1,2,3])

simulator.Initialize(init_vertices, init_elements, 1e3, 1e4, 0.3)
simulator.position[3] = ti.Vector([0.0, 0.0, 2.0])

element_np = simulator.undeformed.elements.to_numpy()
folder = Path("./") / "basic_test"
create_folder(folder, exist_ok=True)
np.save(folder / "elements.npy", element_np)
position_0 = simulator.position.to_numpy()
np.save(folder / "0000.npy", position_0)

for i in tqdm(range(2)):
    position_np = simulator.position.to_numpy()
    print(f"current step {i}, current position {position_np}")
    simulator.Forward(0.01)
    position_np = simulator.position.to_numpy()
    np.save(folder / "{:04d}.npy".format(i + 1), position_np)
# EPSILON = 2e-3
# test_pos = ti.Matrix([[0.0, 0.0, 10.0], [1.0, 0.0, 10.0], [0.0, 1.0, 10.0], [0.0, 0.0, 11.0]])
# test_pos_1 = ti.Matrix([[0.0, 0.0, 10.0], [1.0, 0.0, 10.0], [0.0, 1.0, 10.0], [0.0, 0.0, 11.0]])
# test_pos_1[0,0] += EPSILON
# e1 = 0
# e2 = 0

# @ti.kernel
# def test():
#     e1 = simulator.ComputeEnergy(test_pos, 0.1)
#     e2 = simulator.ComputeEnergy(test_pos_1, 0.1)
#     print((e2 - e1) / EPSILON)
#     simulator.ComputeEnergyGradient(test_pos, 0.1)

# test()
# print(simulator.energy_gradient)


    

# finiteElementTypeDict = dict(
#     vertices_num = ti.i32,
#     vertices = ti.types.matrix(4,3,dtype=ti.f32),
#     polynomials = ti.types.matrix(4,4,dtype=ti.f32),
#     geometry_info = ti.types.struct(
#         dim=ti.i32,
#         vertices_num=ti.i32,
#         vertex_indices=ti.types.vector(4, ti.i32),
#         measure = ti.f32,
#     )
#     # vertices = ti.Vector.field(n=3, dtype=ti.f32, shape=4),
#     # polynomials = ti.Vector.field(n=4, dtype=ti.f32, shape=4),
#     # geometry_info = ti.Struct.field({
#     #     'dim': ti.i32,
#     #     'vertices_num': ti.i32,
#     #     'vertex_indices': ti.types.vector(4, ti.i32),
#     #     'measure': ti.f32,
#     # }, shape=(4,6)),
# )



# finiteElements = ti.Struct.field(
#     finiteElementTypeDict,
#     shape=4
# )