import taichi as ti
from finite_element import FiniteElement
from finite_element import GeometryShape

if __name__ == '__main__':
    ti.init(arch=ti.cuda)

@ti.data_oriented
class Domain:
    @ti.kernel
    def __init__(self):
        self.vertices_num = 4
        self.elements_num = 1
        self.vertices = ti.Vector.field(n=3, dtype=ti.f32, shape=self.vertices_num)
        self.elements = ti.Vector.field(n=4, dtype=ti.int32, shape=self.elements_num)
        self.finite_elements = ti.Struct.field(FiniteElement, shape=self.elements_num)
        self.geometry_info = ti.Struct.field(GeometryShape, shape=(4, self.elements_num))


    @ti.func
    def Initialize(self, vertices, elements):
        self.vertices = vertices
        self.elements = elements
        self.vertices_num = vertices.shape[0]
        self.elements_num = elements.shape[0]
        self.finite_elements = ti.Struct.field(FiniteElement, shape=self.elements_num)
        self.geometry_info = ti.Struct.field(GeometryShape, shape=(4, 6 * self.elements_num))
        for i in range(self.elements_num):
            self.finite_elements[i].Initialize(self.elements[i, 0], self.elements[i, 1], self.elements[i, 2], self.elements[i, 3])

        # Set geometry info
        reverse_map = {}
        # 0D
        for i in range(self.vertices_num):
            self.geometry_info[0, i].dim = 0
            self.geometry_info[0, i].vertices_num = 1
            self.geometry_info[0, i].vertex_indicies = i
            self.geometry_info[0, i].measure = 0.0
              
