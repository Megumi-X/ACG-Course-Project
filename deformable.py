import taichi as ti
import sympy as sp
import scipy
import numpy as np

from domain import Domain, IntegratePoly
from material import Material
from material import materialTypeDict, ComputeEnergyDensity, ComputeStressDensity
from copy import deepcopy
import os

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
        #self.int_matrix = ti.field(dtype=ti.f32)
        #self.int_density_matrix = ti.field(dtype=ti.f32)
        self.int_matrix = ti.Matrix.field(n=vertices_num, m=vertices_num, dtype=ti.f32, shape=())
        self.int_density_matrix = ti.Matrix.field(n=vertices_num, m=vertices_num, dtype=ti.f32, shape=())
        block_size = 4
        grid_size = (vertices_num + block_size - 1) // block_size
        #ti.root.pointer(ti.ij, grid_size).dense(ti.ij, block_size).place(self.int_matrix)
        #ti.root.pointer(ti.ij, grid_size).dense(ti.ij, block_size).place(self.int_density_matrix)
        self.elastic_gradient_map = ti.Matrix.field(n=20, m=2, dtype=ti.i32, shape=vertices_num)
        self.vertices_num = vertices_num
        self.element_num = element_num
        self.free_vertex = ti.Matrix([[0, 0, 0] for i in range(self.vertices_num)])
        self.free_vertex_vector_field = ti.Vector.field(n=3, dtype=ti.i32, shape=vertices_num)
        self.dirichlet_vertex = ti.Matrix([[0, 0, 0] for i in range(self.vertices_num)])
        self.dirichlet_value = ti.Matrix([[0.0, 0.0, 0.0] for i in range(self.vertices_num)])
        self.h = ti.field(dtype=ti.float32, shape=())
        self.x0_np = ti.Matrix.field(n=vertices_num, m=3, dtype=ti.f32, shape=())
        
        # np.array([0 for i in range(self.vertices_num * 3)])
        self.x0_next_np = ti.Matrix.field(n=vertices_num, m=3, dtype=ti.f32, shape=())
        self.res = ti.Matrix.field(n=vertices_num, m=3, dtype=ti.f32, shape=())
        self.elastic_gradient = ti.Matrix.field(n=3, m=4, dtype=ti.f32, shape=self.undeformed.elements_num)

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
                #print(finite_element.polynomials[i, 0])
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
                    #print(value)
                    # w_ij = IntegratePoly(poly_phi_i * poly_phi_j, finite_element.vertices, finite_element.geometry_info_measure)
                    # w_ij = finite_element.IntegratePoly(poly_phi_i * poly_phi_j)
                    if i == j:
                        local_matrix[i, j] += w_ij
                        # print("AAAAAAAAAAAAAA")
                        # print(element)
                        #ti.activate(self.int_matrix, [i, i])
                        self.int_matrix[None][int(element[i]), int(element[i])] += w_ij
                        #ti.activate(self.int_density_matrix,[i, i])
                        self.int_density_matrix[None][element[i], element[i]] += w_ij * self.material[e].density
                    else:
                        local_matrix[i, j] = w_ij
                        local_matrix[j, i] = w_ij
                        #ti.activate(self.int_matrix, [i, j])
                        self.int_matrix[None][element[i], element[j]] += w_ij
                        #ti.activate(self.int_matrix, [j, i])
                        self.int_matrix[None][element[j], element[i]] += w_ij
                        #ti.activate(self.int_density_matrix, [i, j])
                        self.int_density_matrix[None][element[i], element[j]] += w_ij * self.material[e].density
                        #ti.activate(self.int_density_matrix, [j, i])
                        self.int_density_matrix[None][element[j], element[i]] += w_ij * self.material[e].density


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
        for e in range(self.vertices_num):
            for i in range(20):
                for j in range(2):
                    self.elastic_gradient_map[e][i, j] = 999999999
        for e in range(elements_num):
            # finite_element = self.undeformed.finite_elements[e]
            element = self.undeformed.elements[e]
            for i in range(4):
                num = 0
                for j in range(20):
                    if self.elastic_gradient_map[element[i]][j, 0] == 999999999:
                        num = j
                        break
                self.elastic_gradient_map[element[i]][j, 0] = e
                self.elastic_gradient_map[element[i]][j, 1] = i
        for i in range(self.vertices_num):
            for d in range(3):
                if self.dirichlet_boundary_condition[i][d] == float('inf'):
                    self.free_vertex[i, d] = 1
                    self.free_vertex_vector_field[i][d] = 1
                    self.dirichlet_vertex[i, d] = 0
                else:
                    self.free_vertex[i, d] = 0
                    self.free_vertex_vector_field[i][d] = 0
                    self.dirichlet_vertex[i, d] = 1
                    self.dirichlet_value[i, d] = self.dirichlet_boundary_condition[i][d]

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
                    basis_derivatives_q[i, j] = finite_element.polynomials[i, j]
            for i in range(3):
                for j in range(4):
                    local_position[i, j] = position[element[j], i]
            F = local_position @ basis_derivatives_q
            energy += ComputeEnergyDensity(F,self.material[e].lam,self.material[e].mu)* finite_element.geometry_info_measure[3, 0]
            # print("energy density: ", ComputeEnergyDensity(F,self.material[e].lam,self.material[e].mu))
            # energy += self.material[e].ComputeEnergyDensity(F) * finite_element.geometry_info[3][0].measure
        return energy
    
    @ti.func
    def ComputeElasticForce(self, position):
        element_num = self.undeformed.elements_num
        vertices_num = self.undeformed.vertices_num
        for e in range(element_num):
            finite_element = self.undeformed.finite_elements[e]
            element = self.undeformed.elements[e]
            basis_derivatives_q = ti.Matrix([[0.0 for i in range(3)] for j in range(4)])
            local_position = ti.Matrix([[0.0 for i in range(4)] for j in range(3)])
            F = ti.Matrix([[0.0 for i in range(3)] for j in range(3)])
            for i in range(4):
                for j in range(3):
                    basis_derivatives_q[i, j] = finite_element.polynomials[i, j]
            for i in range(3):
                for j in range(4):
                    local_position[i, j] = position[element[j], i]
            F = local_position @ basis_derivatives_q
            P = ComputeStressDensity(F,self.material[e].lam,self.material[e].mu)
            # P = self.material[e].ComputeStressDensity(F)
            self.elastic_gradient[e] = P @ basis_derivatives_q.transpose() * finite_element.geometry_info_measure[3, 0]
        force = ti.Matrix([[0.0 for i in range(3)] for j in range(self.vertices_num)])
        ti.loop_config(serialize=True)
        for i in (range(self.vertices_num)):
            num = 0
            for j in range(20):
                if self.elastic_gradient_map[i][j, 0] == 999999999:
                    num = j
                    break
            for p in range(num):
                e = self.elastic_gradient_map[i][p, 0]
                j = self.elastic_gradient_map[i][p, 1]
                force[i, 0] += -self.elastic_gradient[e][0, j]
                force[i, 1] += -self.elastic_gradient[e][1, j]
                force[i, 2] += -self.elastic_gradient[e][2, j]
        return force

    @ti.func
    def ComputeEnergy(self, position, time_step) -> float:
        vertices_num = self.vertices_num
        inv_h = 1 / time_step
        y = ti.Matrix([[0.0 for i in range(3)] for j in range(self.vertices_num)])
        for i in range(vertices_num):
            for d in range(3):
                y[i, d] = self.position[i][d] + self.velocity[i][d] * time_step + self.external_acceleration[i][d] * time_step * time_step
        coefficient = inv_h * inv_h / 2 
        kinetic_energy = 0.0
        delta_0 = ti.Vector([0.0 for i in range(self.vertices_num)])
        delta_1 = ti.Vector([0.0 for i in range(self.vertices_num)])
        delta_2 = ti.Vector([0.0 for i in range(self.vertices_num)])
        for i in range(vertices_num):
            x_next_0 = position[i,0]
            y_0 = y[i, 0]
            x_next_1 = position[i,1]
            y_1 = y[i, 1]
            x_next_2 = position[i,2]
            y_2 = y[i, 2]
            delta_0[i] = x_next_0 - y_0
            delta_1[i] = x_next_1 - y_1
            delta_2[i] = x_next_2 - y_2
        kinetic_energy += delta_0.dot(self.int_density_matrix[None] @ delta_0)
        kinetic_energy += delta_1.dot(self.int_density_matrix[None] @ delta_1)
        kinetic_energy += delta_2.dot(self.int_density_matrix[None] @ delta_2)
        kinetic_energy *= coefficient
        elastic_energy = self.ComputeElasticEnergy(position)
        print("kinetic_energy: ", kinetic_energy)
        print("elastic_energy: ", elastic_energy)
        return kinetic_energy + elastic_energy
    
    @ti.func
    def ComputeEnergyGradient(self, position, time_step):
        vertices_num = self.vertices_num
        inv_h = 1 / time_step
        coefficient = inv_h * inv_h
        y = ti.Matrix([[0.0 for i in range(3)] for j in range(self.vertices_num)])
        for i in range(vertices_num):
            for d in range(3):
                y[i, d] = self.position[i][d] + self.velocity[i][d] * time_step + self.external_acceleration[i][d] * time_step * time_step
        kinetic_gradient = ti.Matrix([[0.0 for i in range(3)] for j in range(self.vertices_num)])
        delta = ti.Matrix([[0.0 for i in range(3)] for j in range(self.vertices_num)])
        for i in range(vertices_num):
            for dd in range(3):
                x_next_d = position[i,dd]
                y_d = y[i, dd]
                #print("x_next_d: ", x_next_d)
                #print("y_d: ", y_d)
                delta[i, dd] = x_next_d - y_d
        kinetic_gradient = self.int_density_matrix[None] @ delta
        kinetic_gradient *= coefficient
        elastic_gradient = -self.ComputeElasticForce(position)
        #print("kinetic_gradient: ", kinetic_gradient)
        #print("elastic_gradient: ", elastic_gradient)
        return (kinetic_gradient + elastic_gradient)

    @ti.func
    def E(self, x_next):
        x_next_ti = ti.field(dtype=ti.f32, shape=(self.vertices_num, 3))
        for i in range(self.vertices_num * 3):
            q = i // 3
            r = i % 3
            x_next_ti[q, r] = x_next[i]
        energy = self.ComputeEnergy(x_next_ti, self.h[None])
        return float(energy)
    
    @ti.func
    def E_gradient(self, x_next):
        x_next_ti = ti.field(dtype=ti.f32, shape=(self.vertices_num, 3))
        for i in range(self.vertices_num * 3):
            q = i // 3
            r = i % 3
            x_next_ti[q, r] = x_next[i]
        g = self.ComputeEnergyGradient(x_next, self.h[None])
        g1 = g * self.free_vertex
        g1_np = np.array([0 for i in range(3 * self.vertices_num)])
        for i in range(self.vertices_num):
            for d in range(3):
                g1_np[i * 3 + d] = g1[i, d]
        return g1_np
    
    @ti.func
    def Optimizer(self):
        
        # res = scipy.optimize.minimize(self.E, self.x0_np, method="L-BFGS-B", jac=self.E_gradient, options={ "ftol": ftol, "maxiter": max_iter, "iprint": -1 })
        print("x0_np: ", self.x0_np[None][0,0])
        return self.minimizer()
        

    @ti.func
    def minimizer(self):
        print("init[0][0]: ", self.x0_np[None][0,0])
        options = dict(
            ftol = 1e-10,
            maxiter = 20,
        )
        ftol = options["ftol"]
        maxiter = options["maxiter"]
        # assert method in ["Adam"]
        b1 = 0.9
        b2 = 0.99
        lr = 1e-3
        m = self.ComputeEnergyGradient(self.x0_np[None],self.h[None])
        v = m**2
        old_pos = self.x0_np[None]
        new_pos = self.x0_np[None]
        success = False
        ti.loop_config(serialize=True)
        for _ in range(maxiter):
            old_E = self.ComputeEnergy(old_pos, self.h[None]) 
            # changed here
            m += -m + b1 * m + (1-b1) * self.ComputeEnergyGradient(new_pos,self.h[None])
            v += -v + b2 * v + (1-b2) * self.ComputeEnergyGradient(new_pos,self.h[None])**2
            ##
            new_pos -= lr*m / (v**0.5 + 1e-13)
            new_E = self.ComputeEnergy(new_pos, self.h[None]) 
            old_pos -= lr*m / (v**0.5 + 1e-13)
            if abs(new_E-old_E)/abs(old_E) < ftol:
                success = True
                break
            if abs(new_E-old_E) < ftol:
                success = True
                break
        if not success:
            print("Failed to converge")
        return new_pos

    
    
    @ti.kernel
    def Forward(self, time_step: ti.f32):
        print(self.position[0])
        for i in ti.ndrange(self.vertices_num):
            for d in ti.ndrange(3):
                self.x0_np[None][i,d] = self.position[i][d]
                self.x0_next_np[None][i,d] = self.next_position[i][d]
        print(self.position[0])
        print(self.x0_np[None])
        self.kernel_Forward(time_step)
        
    @ti.func
    def kernel_Forward(self, time_step: ti.f32):
        # print(self.position)
        self.h[None] = time_step  
        
          
        # Optimize
        '''
        x0_np = np.array([[self.position[i][j] for j in range(3)] for i in range(self.vertices_num)])
        x_next_np = x0_np
        max_iter = 100
        ftol = 1e-10
        def callback(OptimizeResult):
            pass
        res = scipy.optimize.minimize(self.E, x0_np, method="L-BFGS-B", jac=self.E_gradient, callback=callback, options={ "ftol": ftol, "maxiter": max_iter, "iprint": -1 })
        for i in range(self.vertices_num):
            for d in range(3):
                if self.free_vertex[i, d] == 1:
                    self.next_position[i][d] = res.x[i, d]
                    self.next_velocity[i][d] = (self.next_position[i][d] - self.position[i][d]) * inv_h
                else:
                    self.next_position[i][d] = self.dirichlet_value[i, d]
                    self.next_velocity[i][d] = 0.0
        '''
        print("x0_np: ", self.x0_np[None][0,0])
        res = self.Optimizer()
        inv_h = 1 / self.h[None]
        for i in range(self.vertices_num):
            for d in range(3):
                if self.free_vertex_vector_field[i][d] == 1:
                    self.next_position[i][d] = res[i, d]
                    self.next_velocity[i][d] = (self.next_position[i][d] - self.position[i][d]) * inv_h
                else:
                    self.next_position[i][d] = self.dirichlet_boundary_condition[i][d]
                    self.next_velocity[i][d] = 0.0