import numpy as np
import taichi as ti
ti.init(arch=ti.cpu)
from deformable import DeformableSimulator
simulator = DeformableSimulator(4,2)
init_vertices = ti.Vector.field(n=3,dtype=ti.f32,shape=4)
init_vertices_vector = ti.Vector([[0.0,1.,0.],[0.0,0.0,0.],[1.,0.0,0.],[1.,1.,0.],[0.0,1.,1.],[0.0,0.0,1.],[1.,0.0,1.],[1.,1.,1.],[2.,2.,3.],[2.,3.,3.]])

# for idx in range(10):
#     for j in range(3):
#         init_vertices[idx,j] = init_vertices_vector[idx,j]
init_vertices = init_vertices_vector

init_elements = ti.Vector([[[1.,0.,0.],[0.,0.,0.],[0.,1.,0.],[0.,0.,1.]],
                    [[1.,0.,5.],[0.,0.,5.],[0.,1.,5.],[0.,0.,6.]]])
init_elements_int = ti.Vector([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

simulator.Initialize(init_vertices, init_elements_int, 1.0, 1.0, 1.0)
simulator.Forward(0.1)
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