import taichi as ti
from finite_element import FiniteElement
from finite_element import GeometryShape

@ti.data_oriented
class Domain:
    def __init__(self, vertices_num: ti.int32, elements_num: ti.int32):
        self.vertices_num = vertices_num
        self.elements_num = elements_num
        self.vertices = ti.Vector.field(n=3, dtype=ti.f32, shape=self.vertices_num)
        self.elements = ti.Vector.field(n=4, dtype=ti.int32, shape=self.elements_num)
        self.finite_elements = ti.Struct.field(FiniteElement, shape=self.elements_num)
        self.geometry_info = ti.Struct.field(GeometryShape, shape=(4, 6 * self.elements_num))


    def Initialize(self, vertices, elements):
        self.vertices = vertices
        self.elements = elements
        for i in range(self.elements_num):
            self.finite_elements[i].Initialize(self.elements[i, 0], self.elements[i, 1], self.elements[i, 2], self.elements[i, 3])

        # Set geometry info
        # 0D
        for i in range(self.vertices_num):
            self.geometry_info[0, i].dim = 0
            self.geometry_info[0, i].vertices_num = 1
            self.geometry_info[0, i].vertex_indicies = i
            self.geometry_info[0, i].measure = 0.0
              
