import taichi as ti
import sympy as sp

from domain import Domain
from material import Material


ti.init(arch=ti.cuda)

@ti.data_oriented
class DeformableSimulator:
    def __init__(self, vertices_num: ti.int32, element_num: ti.int32):
        self.undeformed = Domain(vertices_num, element_num)
        self.material = ti.Struct.field(Material, shape=element_num)
        self.position = ti.Vector.field(n=3, dtype=ti.f32, shape=vertices_num)
        self.pre_position = ti.Vector.field(n=3, dtype=ti.f32, shape=vertices_num)
        self.velocity = ti.Vector.field(n=3, dtype=ti.f32, shape=vertices_num)
        self.pre_velocity = ti.Vector.field(n=3, dtype=ti.f32, shape=vertices_num)
        self.next_position = ti.Vector.field(n=3, dtype=ti.f32, shape=vertices_num)
        self.next_velocity = ti.Vector.field(n=3, dtype=ti.f32, shape=vertices_num)
        self.external_acceleration = ti.Vector.field(n=3, dtype=ti.f32, shape=vertices_num)
        self.dirichlet_boundary_condition = ti.Vector.field(n=3, dtype=ti.f32, shape=vertices_num)
        self.int_matrix = ti.field(dtype=ti.f32)
        self.int_density_matrix = ti.field(dtype=ti.f32)
        block_size = 4
        grid_size = (vertices_num + block_size - 1) // block_size
        ti.root.pointer(ti.ij, grid_size).dense(ti.ij, block_size).place(self.int_matrix)
        ti.root.pointer(ti.ij, grid_size).dense(ti.ij, block_size).place(self.int_density_matrix)

    @ti.kernel
    def InitializePosition(self):
        vertices_num = self.undeformed.vertices_num
        for i in range(vertices_num):
            self.position[i] = self.undeformed.vertices[i]
            self.pre_position[i] = self.undeformed.vertices[i]
            self.velocity[i] = ti.Vector([0.0, 0.0, 0.0])
            self.pre_velocity[i] = ti.Vector([0.0, 0.0, 0.0])
            self.next_position[i] = ti.Vector([0.0, 0.0, 0.0])
            self.next_velocity[i] = ti.Vector([0.0, 0.0, 0.0])
            self.external_acceleration[i] = ti.Vector([0.0, 0.0, 0.0])
            self.dirichlet_boundary_condition[i] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def ComputeIntMatrix(self):
        elements_num = self.undeformed.elements_num
        for e in range(elements_num):
            finite_element = self.undeformed.finite_elements[e]
            element = self.undeformed.elements[e]
            local_matrix = ti.Matrix([[0.0 for i in range(4)] for j in range(4)])
            for i in range(4):
                phi_i = finite_element.polynomials[i]
                x, y, z = sp.symbols('x y z')
                poly_phi_i = phi_i[0] * x + phi_i[1] * y + phi_i[2] * z + phi_i[3] 
                for j in range(i, 4):
                    phi_j = finite_element.polynomials[j]
                    poly_phi_j = phi_j[0] * x + phi_j[1] * y + phi_j[2] * z + phi_j[3] 
                    w_ij = finite_element.IntegratePoly(poly_phi_i * poly_phi_j)
                    if i == j:
                        local_matrix[i, j] = w_ij
                        ti.activate(self.int_matrix, element[i], element[i])
                        self.int_matrix[element[i], element[i]] = w_ij
                        ti.activate(self.int_density_matrix, element[i], element[i])
                        self.int_density_matrix[element[i], element[i]] = w_ij * self.material[e].density
                    else:
                        local_matrix[i, j] = w_ij
                        local_matrix[j, i] = w_ij
                        ti.activate(self.int_matrix, element[i], element[j])
                        self.int_matrix[element[i], element[j]] = w_ij
                        ti.activate(self.int_matrix, element[j], element[i])
                        self.int_matrix[element[j], element[i]] = w_ij
                        ti.activate(self.int_density_matrix, element[i], element[j])
                        self.int_density_matrix[element[i], element[j]] = w_ij * self.material[e].density
                        ti.activate(self.int_density_matrix, element[j], element[i])
                        self.int_density_matrix[element[j], element[i]] = w_ij * self.material[e].density


    def Initialize(self, vertices, elements, density, youngs_modulus, poissons_ratio):
        self.undeformed.Initialize(vertices, elements)

        # Initialize the material
        elements_num = self.undeformed.elements_num
        for i in range(elements_num):
            self.material[i].Initialize(density, youngs_modulus, poissons_ratio)

        # Initialize the position and velocity
        vertices_num = self.undeformed.vertices_num
        self.InitializePosition()     

        # Compute the integration matrix
        self.ComputeIntMatrix()