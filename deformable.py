import taichi as ti
import sympy as sp
import scipy
import numpy as np

from domain import Domain, IntegratePoly
from material import Material
from material import materialTypeDict, ComputeEnergyDensity, ComputeStressDensity


ti.init(arch=ti.cuda)

@ti.data_oriented
class DeformableSimulator:
    def __init__(self, vertices_num: ti.int32, element_num: ti.int32):
        self.undeformed = Domain(vertices_num, element_num)
        self.material = ti.Struct.field(materialTypeDict, shape=element_num)
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
        self.elastic_gradient_map = []

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
            self.dirichlet_boundary_condition[i] = ti.Vector([float('inf'), float('inf'), float('inf')])

    @ti.kernel
    def ComputeIntMatrix(self):
        elements_num = self.undeformed.elements_num
        for e in range(elements_num):
            finite_element = self.undeformed.finite_elements[e]
            element = self.undeformed.elements[e]
            local_matrix = ti.Matrix([[0.0 for i in range(4)] for j in range(4)])
            for i in range(4):
                phi_i = ti.Vector([finite_element.polynomials[i,j] for j in range(6)])
                # x, y, z = sp.symbols('x y z')
                # poly_phi_i = phi_i[0] * x + phi_i[1] * y + phi_i[2] * z + phi_i[3] 
                for j in range(i, 4):
                    phi_j = ti.Vector([finite_element.polynomials[j,jj] for jj in range(6)])
                    # poly_phi_j = phi_j[0] * x + phi_j[1] * y + phi_j[2] * z + phi_j[3] 
                    pos = ti.Vector([0.0,0.0,0.0])
                    for ii in range(4):
                        pos += ti.Vector([finite_element.vertices[ii,jj] for jj in range(3)])
                    # pos = (finite_element.vertices[0] + finite_element.vertices[1] + finite_element.vertices[2] + finite_element.vertices[3]) / 4
                    value = (phi_i[0] * pos[0] + phi_i[1] * pos[1] + phi_i[2] * pos[2] + phi_i[3]) * (phi_j[0] * pos[0] + phi_j[1] * pos[1] + phi_j[2] * pos[2] + phi_j[3])
                    w_ij = value * finite_element.geometry_info_measure[3,0]
                    # w_ij = IntegratePoly(poly_phi_i * poly_phi_j, finite_element.vertices, finite_element.geometry_info_measure)
                    # w_ij = finite_element.IntegratePoly(poly_phi_i * poly_phi_j)
                    if i == j:
                        local_matrix[i, j] += w_ij
                        # print("AAAAAAAAAAAAAA")
                        # print(element)
                        #ti.activate(self.int_matrix, [i, i])
                        self.int_matrix[element[i], element[i]] += w_ij
                        #ti.activate(self.int_density_matrix,[i, i])
                        self.int_density_matrix[element[i], element[i]] += w_ij * self.material[e].density
                    else:
                        local_matrix[i, j] = w_ij
                        local_matrix[j, i] = w_ij
                        #ti.activate(self.int_matrix, [i, j])
                        self.int_matrix[element[i], element[j]] += w_ij
                        #ti.activate(self.int_matrix, [j, i])
                        self.int_matrix[element[j], element[i]] += w_ij
                        #ti.activate(self.int_density_matrix, [i, j])
                        self.int_density_matrix[element[i], element[j]] += w_ij * self.material[e].density
                        #ti.activate(self.int_density_matrix, [j, i])
                        self.int_density_matrix[element[j], element[i]] += w_ij * self.material[e].density


    def Initialize(self, vertices, elements, density, youngs_modulus, poissons_ratio):
        self.undeformed.Initialize(vertices, elements)

        # Initialize the material
        elements_num = self.undeformed.elements_num
        for i in range(elements_num):
            self.material[i].density = density
            self.material[i].youngs_modulus = youngs_modulus
            self.material[i].poissons_ratio = poissons_ratio
            self.material[i].lam = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
            self.material[i].mu = youngs_modulus / (2 * (1 + poissons_ratio))
            # self.material[i].Initialize(density, youngs_modulus, poissons_ratio)

        # Initialize the position and velocity
        vertices_num = self.undeformed.vertices_num
        self.InitializePosition()     

        # Compute the integration matrix
        self.ComputeIntMatrix()

        # Build the elastic gradient map
        self.elastic_gradient_map.clear()
        self.elastic_gradient_map = [[] for i in range(vertices_num)]
        for e in range(elements_num):
            # finite_element = self.undeformed.finit   e_elements[e]
            element = self.undeformed.elements[e]
            self.elastic_gradient_map[e] = []
            for i in range(4):
                self.elastic_gradient_map[element[i]].append([e, i])

    @ti.func
    def ComputeElasticEnergy(self, position):
        element_num = self.undeformed.elements_num
        energy = 0.0
        for e in range(element_num):
            finite_element = self.undeformed.finite_elements[e]
            element = self.undeformed.elements[e]
            basis_derivatives_q = ti.Matrix([[0.0 for i in range(3)] for j in range(4)])
            local_position = ti.Matrix([[0.0 for i in range(4)] for j in range(3)])
            F = ti.Matrix([[0.0 for i in range(3)] for j in range(3)])
            for i in range(4):
                for j in range(3):
                    basis_derivatives_q[i, j] = finite_element.polynomials[i][j]
            for i in range(3):
                for j in range(4):
                    local_position[i, j] = position[i, element[j]]
            F = local_position @ basis_derivatives_q
            energy += ComputeEnergyDensity(F,self.material[e].lam,self.material[e].mu)* finite_element.geometry_info[3][0].measure
            # energy += self.material[e].ComputeEnergyDensity(F) * finite_element.geometry_info[3][0].measure
        return energy
    
    @ti.func
    def ComputeElasticForce(self, position):
        element_num = self.undeformed.elements_num
        vertices_num = self.undeformed.vertices_num
        gradient = ti.Matrix.field(n=3, m=4, dtype=ti.f32, shape=self.undeformed.elements_num)
        for e in range(element_num):
            finite_element = self.undeformed.finite_elements[e]
            element = self.undeformed.elements[e]
            basis_derivatives_q = ti.Matrix([[0.0 for i in range(3)] for j in range(4)])
            local_position = ti.Matrix([[0.0 for i in range(4)] for j in range(3)])
            F = ti.Matrix([[0.0 for i in range(3)] for j in range(3)])
            for i in range(4):
                for j in range(3):
                    basis_derivatives_q[i, j] = finite_element.polynomials[i][j]
            for i in range(3):
                for j in range(4):
                    local_position[i, j] = position[i, element[j]]
            F = local_position @ basis_derivatives_q
            P = ComputeStressDensity(F,self.material[e].lam,self.material[e].mu)
            # P = self.material[e].ComputeStressDensity(F)
            gradient[e] = P @ basis_derivatives_q.transpose() * finite_element.geometry_info[3][0].measure
        force = ti.Matrix(n=vertices_num, m=3, dtype=ti.f32)
        for i in range(vertices_num):
            for pair in self.elastic_gradient_map[i]:
                e = pair[0]
                j = pair[1]
                force[i, 0] += -gradient[e][0, j]
                force[i, 1] += -gradient[e][1, j]
                force[i, 2] += -gradient[e][2, j]
        return force

    @ti.func
    def ComputeEnergy(self, position, time_step):
        h = time_step
        vertices_num = self.undeformed.vertices_num
        inv_h = 1 / h
        x0 = self.position
        v0 = self.velocity
        x_next = position
        a = self.external_acceleration
        y = x0 + v0 * h + a * h * h
        coefficient = inv_h * inv_h / 2 
        kinetic_energy = 0.0
        delta = ti.Vector.field(n=vertices_num, dtype=ti.f32, shape=3)
        for i in range(vertices_num):
            for d in range(3):
                x_next_d = x_next[i, d]
                y_d = y[i, d]
                delta[d, i] = x_next_d - y_d
        for d in range(3):
            kinetic_energy += delta[d].dot(self.int_density_matrix @ delta[d])
        kinetic_energy *= coefficient
        elastic_energy = self.ComputeElasticEnergy(position)
        return kinetic_energy + elastic_energy
    
    @ti.func
    def ComputeEnergyGradient(self, position, time_step):
        h = time_step
        vertices_num = self.undeformed.vertices_num
        inv_h = 1 / h
        x0 = self.position
        v0 = self.velocity
        x_next = position
        a = self.external_acceleration
        y = x0 + v0 * h + a * h * h
        coefficient = inv_h * inv_h
        kinetic_gradient = ti.Matrix(n=vertices_num, m=3, dtype=ti.f32)
        delta = ti.Matrix(n=vertices_num, m=3, dtype=ti.f32)
        for i in range(vertices_num):
            for d in range(3):
                x_next_d = x_next[i, d]
                y_d = y[i, d]
                delta[i, d] = x_next_d - y_d
        kinetic_gradient = self.int_density_matrix @ delta
        kinetic_gradient *= coefficient
        elastic_gradient = -self.ComputeElasticForce(x_next)
        return (kinetic_gradient + elastic_gradient)

        
    @ti.kernel
    def Forward(self, time_step: ti.f32):
        x0 = self.position
        h = time_step
        inv_h = 1 / h
        vertices_num = self.undeformed.vertices_num
        free_vertex = ti.Matrix(n=vertices_num, m=3, dtype=ti.int32)
        dirichlet_vertex = ti.Matrix(n=vertices_num, m=3, dtype=ti.int32)
        dirichlet_value = ti.Matrix(n=vertices_num, m=3, dtype=ti.f32)
        for i in range(vertices_num):
            for d in range(3):
                if self.dirichlet_boundary_condition[i][d] == float('inf'):
                    free_vertex[i, d] = 1
                    dirichlet_vertex[i, d] = 0
                else:
                    free_vertex[i, d] = 0
                    dirichlet_vertex[i, d] = 1
                    dirichlet_value[i, d] = self.dirichlet_boundary_condition[i][d]
        def E(x_next):
            x_next_ti = ti.field(dtype=ti.f32, shape=(vertices_num, 3))
            x_next_ti.from_numpy(x_next)
            return self.ComputeEnergy(x_next_ti, h)
        def E_gradient(x_next):
            x_next_ti = ti.field(dtype=ti.f32, shape=(vertices_num, 3))
            x_next_ti.from_numpy(x_next)
            g = self.ComputeEnergyGradient(x_next, h)
            g1 = g * free_vertex
            g1_np = np.array([[g1[i, j] for j in range(3)] for i in range(vertices_num)])
            return g1_np
        
        # Optimize
        x0_np = np.array([[x0[i, j] for j in range(3)] for i in range(vertices_num)])
        x_next_np = x0_np
        max_iter = 100
        ftol = 1e-10
        def callback(OptimizeResult):
            pass
        res = scipy.optimize.minimize(E, x0_np, method="L-BFGS-B", jac=E_gradient, callback=callback, options={ "ftol": ftol, "maxiter": max_iter, "iprint": -1 })
        for i in range(vertices_num):
            for d in range(3):
                if free_vertex[i, d] == 1:
                    self.next_position[i][d] = res.x[i, d]
                    self.next_velocity[i][d] = (self.next_position[i][d] - self.position[i][d]) * inv_h
                else:
                    self.next_position[i][d] = dirichlet_value[i, d]
                    self.next_velocity[i][d] = 0.0