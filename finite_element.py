import taichi as ti
import sympy as sp

@ti.dataclass
class GeometryShape:
    dim: ti.int32
    vertices_num: ti.int32
    vertex_indices: ti.types.vector(4, ti.int32)
    measure: ti.f64
geometry_info_type = ti.types.struct(
        dim=ti.i32,
        vertices_num=ti.i32,
        vertex_indices=ti.types.vector(4, ti.i32),
        measure = ti.f64,
    )
finiteElementTypeDict = dict(
    vertices_num = ti.i32,
    vertices = ti.types.matrix(4,3,dtype=ti.f64),
    polynomials = ti.types.matrix(4,4,dtype=ti.f64),
    geometry_info_dim = ti.types.matrix(4,6,dtype=ti.i32),
    geometry_info_vertices_num = ti.types.matrix(4,6,dtype=ti.i32),
    geometry_info_vertex_indices_0 = ti.types.matrix(6,4,ti.i32),
    geometry_info_vertex_indices_1 = ti.types.matrix(6,4,ti.i32),
    geometry_info_vertex_indices_2 = ti.types.matrix(6,4,ti.i32),
    geometry_info_vertex_indices_3 = ti.types.matrix(6,4,ti.i32),
    geometry_info_measure = ti.types.matrix(4,6,ti.f64),
)
@ti.data_oriented
class FiniteElement:
    def __init__(self):
        self.vertices_num = 4
        self.vertices = ti.Vector.field(n=3, dtype=ti.f64, shape=4)
        self.polynomials = ti.Vector.field(n=4, dtype=ti.f64, shape=4)
        self.geometry_info = ti.Struct.field({
            'dim': ti.i32,
            'vertices_num': ti.i32,
            'vertex_indices': ti.types.vector(4, ti.i32),
            'measure': ti.f64,
        }, shape=(4,6))
    @ti.kernel
    def Initialize(self, v0: ti.types.vector(3, ti.f64), v1: ti.types.vector(3, ti.f64), v2: ti.types.vector(3, ti.f64), v3: ti.types.vector(3, ti.f64)):
        # Set vertices
        self.vertices[0] = v0
        self.vertices[1] = v1
        self.vertices[2] = v2
        self.vertices[3] = v3

        # Set geometry info
        volume = (v1 - v0).cross(v2 - v1).dot(v3 - v2) / 6
        if volume < 0 :
            order = ti.Vector([0, 1, 2, 3])
            volume = - volume
        else:
            order = ti.Vector([0, 2, 1, 3])
        # 0D
        for i in range(4): 
            self.geometry_info[0, i].dim = 0
            self.geometry_info[0, i].vertices_num = 1
            self.geometry_info[0, i].vertex_indicies = i
            self.geometry_info[0, i].measure = 0.0
        edge_index = 0
        # 1D
        for i in range(4): 
            for j in range(i + 1, 4):
                self.geometry_info[1, edge_index].dim = 1
                self.geometry_info[1, edge_index].vertices_num = 2
                self.geometry_info[1, edge_index].vertex_indicies[0] = i
                self.geometry_info[1, edge_index].vertex_indicies[1] = j
                self.geometry_info[1, edge_index].measure = (self.vertices[i] - self.vertices[j]).norm()
                edge_index += 1
        # 2D
        self.geometry_info[2, 0].vertex_indicies = ti.Vector([order[0], order[1], order[2], 0])
        self.geometry_info[2, 1].vertex_indicies = ti.Vector([order[1], order[0], order[3], 0])
        self.geometry_info[2, 2].vertex_indicies = ti.Vector([order[2], order[1], order[3], 0])
        self.geometry_info[2, 3].vertex_indicies = ti.Vector([order[0], order[2], order[3], 0])
        for i in range(4):
            self.geometry_info[2, i].dim = 2
            self.geometry_info[2, i].vertices_num = 3
            v0 = self.geometry_info[2, i].vertex_indices[0]
            v1 = self.geometry_info[2, i].vertex_indices[1]
            v2 = self.geometry_info[2, i].vertex_indices[2]
            v01 = self.vertices[v1] - self.vertices[v0]
            v02 = self.vertices[v2] - self.vertices[v0]
            self.geometry_info[2, i].measure = v01.cross(v02).norm() / 2
        # 3D
        self.geometry_info[3, 0].vertex_indices = order
        self.geometry_info[3, 0].dim = 3
        self.geometry_info[3, 0].vertices_num = 4
        self.geometry_info[3, 0].measure = volume

        # Set polynomials
        A = ti.Matrix([[0.0 for i in range(self.vertices_num)] for j in range(4)])
        for i in range(4):
            A[0, i] = self.vertices[i][0]
            A[1, i] = self.vertices[i][1]
            A[2, i] = self.vertices[i][2]
            A[3, i] = 1.0
        A_inv = A.inverse()
        for i in range(4):
            self.polynomials[i][0] = A_inv[i, 0]
            self.polynomials[i][1] = A_inv[i, 1]
            self.polynomials[i][2] = A_inv[i, 2]
            self.polynomials[i][3] = A_inv[i, 3]

    @ti.kernel
    def Function(self, values: ti.types.vector(4, ti.f64)) -> ti.types.vector(4, ti.f64):
        ret = ti.Vector([0.0 for i in range(4)])
        for i in range(4):
            ret += self.polynomials[i] * values[i]
        return ret
    
    @ti.func
    def IntegratePoly(self, poly):
        pos = (self.vertices[0] + self.vertices[1] + self.vertices[2] + self.vertices[3]) / 4
        x, y, z = sp.symbols('x y z')
        value = poly.subs([(x, pos[0]), (y, pos[1]), (z, pos[2])]).evalf()
        return value * self.geometry_info[3, 0].measure



# polynomials = ti.Vector.field(n=4, dtype=ti.f64, shape=4).
@ti.kernel
def Function(values:ti.types.vector(4,ti.f64),polynomials:ti.types.matrix(4,4,ti.f64)) -> ti.types.vector(4,ti.f64):
    ret = ti.Vector([0.0 for i in range(4)])
    for i in range(4):
        ret += polynomials[i] * values[i]
    return ret

@ti.func
def IntegratePoly(poly,vertices:ti.types.matrix(4,3,ti.f64),geometry_info_measure: ti.types.matrix(4,6,ti.f64)):
    pos = (vertices[0] + vertices[1] + vertices[2] + vertices[3]) / 4
    x, y, z = sp.symbols('x y z')
    value = poly.subs([(x, pos[0]), (y, pos[1]), (z, pos[2])]).evalf()
    return value * geometry_info_measure[3, 0]