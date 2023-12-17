import taichi as ti
import sympy as sp
import scipy
import numpy as np

from domain import Domain, IntegratePoly
from material import Material
from material import materialTypeDict, ComputeEnergyDensity, ComputeStressDensity

HISTORY_SIZE = 5
N=90
alpha_field = ti.field(dtype=ti.f64, shape=(N))
for i in range(N):
    alpha_field[i] = ((-1)**i)*2.0*(0.8)**i*1e-4
abs_alpha_field = ti.field(dtype = ti.f64,shape=(N))
for i in range(N):
    abs_alpha_field[i] = ((-1)**i)*2*(0.8)**i

@ti.func
def copy_fields(src: ti.template(), dest: ti.template()):
    for I in ti.grouped(src):
        dest[I] = src[I]

@ti.func
def copy_fields_square(src: ti.template(), dest: ti.template()):
    for I in ti.grouped(src):
        dest[I] = src[I]**2








@ti.func
def norm(x:ti.template()):
    sum_ = 0.0
    for I in ti.grouped(x):
        sum_ += x[I]**2
    return sum_**0.5

@ti.func
def reverse_fields(src:ti.template()):
    for I in ti.grouped(src):
        src[I] = -src[I]

@ti.func
def compute_add_with_mult(add1:ti.template(),add2:ti.template(),alpha:ti.f64,dest:ti.template()):
    for I in ti.grouped(add1):
        dest[I] = add1[I]*alpha + add2[I]
@ti.func
def compute_normed(src:ti.template(),dest:ti.template()):
    EPSILON = 1e-30
    norm_ = norm(src)
    for I in ti.grouped(src):
        dest[I] = src[I]/(norm_+EPSILON)

@ti.func
def compute_difference_norm(src1:ti.template(),src2:ti.template()):
    sum_ = 0.0
    for I in ti.grouped(src1):
        sum_ += (src1[I] - src2[I])**2
    return sum_**0.5


@ti.func
def copy_fields_2d(src:ti.template(), dest:ti.template(), dim0:ti.i32,dim1:ti.i32):
    for i,j in ti.ndrange(dim0,dim1):
        dest[i,j] = src[i,j]

@ti.func
def copy_fields_2d_square(src:ti.template(), dest:ti.template(), dim0:ti.i32,dim1:ti.i32):
    for i,j in ti.ndrange(dim0,dim1):
        dest[i,j] = src[i,j]**2
@ti.func
def norm_2d(x: ti.template(), dim0: ti.i32, dim1: ti.i32):
    sum_ = 0.0
    for i, j in ti.ndrange(dim0, dim1):
        sum_ += x[i, j]**2
    return sum_**0.5

@ti.func
def reverse_fields_2d(src: ti.template(), dim0: ti.i32, dim1: ti.i32):
    for i, j in ti.ndrange(dim0, dim1):
        src[i, j] = -src[i, j]

@ti.func
def compute_add_with_mult_2d(add1: ti.template(), add2: ti.template(), alpha: ti.f64, dest: ti.template(),
                              dim0: ti.i32, dim1: ti.i32):
    for i, j in ti.ndrange(dim0, dim1):
        dest[i, j] = add1[i, j] + add2[i, j] * alpha

@ti.func
def compute_add_with_mult_2d_add_3d(add1: ti.template(), add2: ti.template(), alpha: ti.f64, dest: ti.template(),
                              dim0: ti.i32, dim1: ti.i32, index):
    for i, j in ti.ndrange(dim0, dim1):
        dest[i, j] = add1[i, j] + add2[index, i, j] * alpha


@ti.func
def compute_normed_2d(src: ti.template(), dest: ti.template(), dim0: ti.i32, dim1: ti.i32):
    EPSILON = 1e-30
    norm_ = norm_2d(src, dim0, dim1)
    # print("Current norm_ == ", norm_)
    for i, j in ti.ndrange(dim0, dim1):
        dest[i, j] = src[i, j] / (norm_+EPSILON)

@ti.func
def compute_difference_norm_2d(src1: ti.template(), src2: ti.template(), dim0: ti.i32, dim1: ti.i32):
    sum_ = 0.0
    for i, j in ti.ndrange(dim0, dim1):
        sum_ += (src1[i, j] - src2[i, j])**2
    return sum_**0.5

@ti.func
def copy_fields_3d(src: ti.template(), dest: ti.template(), dim0: ti.i32, dim1: ti.i32, dim2: ti.i32):
    for i, j, k in ti.ndrange(dim0, dim1, dim2):
        dest[i, j, k] = src[i, j, k]

@ti.func
def copy_fields_square_3d(src: ti.template(), dest: ti.template(), dim0: ti.i32, dim1: ti.i32, dim2: ti.i32):
    for i, j, k in ti.ndrange(dim0, dim1, dim2):
        dest[i, j, k] = src[i, j, k]**2
@ti.func
def norm_3d(x: ti.template(), dim0: ti.i32, dim1: ti.i32, dim2: ti.i32):
    sum_ = 0.0
    for i, j, k in ti.ndrange(dim0, dim1, dim2):
        sum_ += x[i, j, k]**2
    return sum_**0.5

@ti.func
def reverse_fields_3d(src: ti.template(), dim0: ti.i32, dim1: ti.i32, dim2: ti.i32):
    for i, j, k in ti.ndrange(dim0, dim1, dim2):
        src[i, j, k] = -src[i, j, k]

@ti.func
def compute_add_with_mult_3d(add1: ti.template(), add2: ti.template(), alpha: ti.f64, dest: ti.template(),
                              dim0: ti.i32, dim1: ti.i32, dim2: ti.i32):
    for i, j, k in ti.ndrange(dim0, dim1, dim2):
        dest[i, j, k] = add1[i, j, k] + add2[i, j, k] * alpha

@ti.func
def compute_normed_3d(src: ti.template(), dest: ti.template(), dim0: ti.i32, dim1: ti.i32, dim2: ti.i32):
    EPSILON = 1e-30
    norm_ = norm_3d(src, dim0, dim1, dim2)
    for i, j, k in ti.ndrange(dim0, dim1, dim2):
        dest[i, j, k] = src[i, j, k] / (norm_+EPSILON)

@ti.func
def compute_difference_norm_3d(src1: ti.template(), src2: ti.template(), dim0: ti.i32, dim1: ti.i32, dim2: ti.i32):
    sum_ = 0.0
    for i, j, k in ti.ndrange(dim0, dim1, dim2):
        sum_ += (src1[i, j, k] - src2[i, j, k])**2
    return sum_**0.5

@ti.func
def sub_dot(src1:ti.template(), src2:ti.template(), index0:ti.i32,dim1:ti.i32,dim2:ti.i32):
    sum_ = 0.0
    for j, k in ti.ndrange(dim1,dim2):
        sum_ += src1[index0,j,k]*src2[index0,j,k]
    return sum_

@ti.func
def dot_2d_and_3d(src1:ti.template(),src2:ti.template(),index0:ti.i32,dim1:ti.i32,dim2:ti.i32):
    sum_ = 0.0
    for j, k in ti.ndrange(dim1,dim2):
        sum_ += src1[j,k] * src2[index0,j,k]
    return sum_



@ti.data_oriented
class DeformableSimulator:
    def __init__(self, vertices_num: ti.int32, element_num: ti.int32):
        self.undeformed = Domain(vertices_num, element_num)
        self.material = ti.Struct.field(materialTypeDict, shape=element_num)
        self.position = ti.Vector.field(n=3, dtype=ti.f64, shape=vertices_num)
        self.pre_position = ti.Vector.field(n=3, dtype=ti.f64, shape=vertices_num)
        self.velocity = ti.Vector.field(n=3, dtype=ti.f64, shape=vertices_num)
        self.pre_velocity = ti.Vector.field(n=3, dtype=ti.f64, shape=vertices_num)
        self.next_position = ti.Vector.field(n=3, dtype=ti.f64, shape=vertices_num)
        self.next_velocity = ti.Vector.field(n=3, dtype=ti.f64, shape=vertices_num)
        self.external_acceleration = ti.Vector.field(n=3, dtype=ti.f64, shape=vertices_num)
        self.dirichlet_boundary_condition = ti.Vector.field(n=3, dtype=ti.f64, shape=vertices_num)
        #self.int_matrix = ti.field(dtype=ti.f64)
        #self.int_density_matrix = ti.field(dtype=ti.f64)
        # self.int_matrix = ti.field(dtype=ti.f64, shape=(vertices_num, vertices_num))
        # self.int_density_matrix = ti.field(dtype=ti.f64, shape=(vertices_num, vertices_num))
        self.int_matrix = ti.field(dtype=ti.f64)
        self.int_density_matrix = ti.field(dtype=ti.f64)
        block_size = 4
        grid_size = (vertices_num + block_size - 1) // block_size
        ti.root.pointer(ti.ij, grid_size).dense(ti.ij, block_size).place(self.int_matrix)
        ti.root.pointer(ti.ij, grid_size).dense(ti.ij, block_size).place(self.int_density_matrix)
        self.elastic_gradient_map = ti.Vector.field(n=2, dtype=ti.i32, shape=(vertices_num, 100))
        self.vertices_num = vertices_num
        self.element_num = element_num
        self.free_vertex = ti.field(dtype=ti.i32, shape=(vertices_num, 3))
        self.free_vertex_vector_field = ti.Vector.field(n=3, dtype=ti.i32, shape=vertices_num)
        self.dirichlet_vertex = ti.field(dtype=ti.i32, shape=(vertices_num, 3))
        self.dirichlet_value = ti.field(dtype=ti.f64, shape=(vertices_num, 3))
        self.h = ti.field(dtype=ti.f64, shape=())
        self.x0_np = ti.field(dtype=ti.f64, shape=(vertices_num, 3))
        self.n = vertices_num
        
        # np.array([0 for i in range(self.vertices_num * 3)])
        self.x0_next_np = ti.field(dtype=ti.f64, shape=(vertices_num, 3))
        self.step_direction = ti.field(dtype=ti.f64, shape=(vertices_num, 3))
        self.step_history = ti.field(dtype=ti.f64,shape = (HISTORY_SIZE,vertices_num,3))
        self.grad_history = ti.field(dtype=ti.f64,shape = (HISTORY_SIZE,vertices_num,3))
        self.g_field = ti.field(dtype=ti.f64, shape=(vertices_num, 3))
        self.q_field = ti.field(dtype=ti.f64, shape=(vertices_num, 3))
        self.x0_np_added = ti.field(dtype=ti.f64, shape=(vertices_num, 3))
        self.step_direction_normed = ti.field(dtype=ti.f64, shape=(vertices_num, 3))
        self.alpha_chosen = ti.field(dtype=ti.f64,shape=())
        self.y_temp = ti.field(dtype=ti.f64, shape=(vertices_num, 3))
        
        self.res = ti.field(dtype=ti.f64, shape=(vertices_num, 3))
        self.elastic_gradient = ti.Matrix.field(n=3, m=4, dtype=ti.f64, shape=self.undeformed.elements_num)

        self.elastic_force = ti.field(dtype=ti.f64, shape=(vertices_num, 3))
        self.delta = ti.field(dtype=ti.f64, shape=(vertices_num, 3))
        self.kinetic_gradient = ti.field(dtype=ti.f64, shape=(vertices_num, 3))
        self.energy_gradient = ti.field(dtype=ti.f64, shape=(vertices_num, 3))
        self.temp_m_field = ti.field(dtype=ti.f64, shape=(vertices_num, 3))
        self.temp_v_field = ti.field(dtype=ti.f64, shape=(vertices_num, 3))
        self.dot_y_q = ti.field(dtype=ti.f64,shape = (HISTORY_SIZE))
        self.dot_y_s = ti.field(dtype=ti.f64,shape = (HISTORY_SIZE))
        self.alpha_field = ti.field(dtype=ti.f64,shape=(HISTORY_SIZE))
        
        self.m_field = ti.field(dtype=ti.f64, shape=(vertices_num, 3))
        self.v_field = ti.field(dtype=ti.f64, shape=(vertices_num, 3))
        self.y = ti.field(dtype=ti.f64, shape=(vertices_num, 3))
        


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
                        self.int_matrix[int(element[i]), int(element[i])] += w_ij
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


    def Initialize(self, vertices:ti.template(), elements:ti.template(), density:ti.f64, youngs_modulus:ti.f64, poissons_ratio:ti.f64):
        self.undeformed.Initialize(vertices, elements)
        print("finish initializing undeformed domain")
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
        # print(self.int_density_matrix)

        # Build the elastic gradient map
        for e in range(self.vertices_num):
            for i in range(100):
                for j in range(2):
                    self.elastic_gradient_map[e, i][j] = 999999999
        for e in range(elements_num):
            # finite_element = self.undeformed.finite_elements[e]
            element = self.undeformed.elements[e]
            for i in range(4):
                num = 0
                for j in range(100):
                    if self.elastic_gradient_map[element[i], j][0] == 999999999:
                        num = j
                        break
                self.elastic_gradient_map[element[i], j][0] = e
                self.elastic_gradient_map[element[i], j][1] = i
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
            # print(F)
            energy += ComputeEnergyDensity(F,self.material[e].lam,self.material[e].mu)* finite_element.geometry_info_measure[3, 0]
            # print("energy density: ", ComputeEnergyDensity(F,self.material[e].lam,self.material[e].mu))
            # energy += self.material[e].ComputeEnergyDensity(F) * finite_element.geometry_info[3][0].measure
        return energy
    
    @ti.func
    def ComputeElasticForce(self, position):
        element_num = self.undeformed.elements_num
        vertices_num = self.undeformed.vertices_num
        for i, j in ti.ndrange(self.vertices_num, 3):
            self.elastic_force[i, j] = 0
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
        #force = ti.Matrix([[0.0 for i in range(3)] for j in range(self.vertices_num)])
        ti.loop_config(serialize=True)
        for i in (range(self.vertices_num)):
            num = 0
            for j in range(100):
                if self.elastic_gradient_map[i, j][0] == 999999999:
                    num = j
                    break
            for p in range(num):
                e = self.elastic_gradient_map[i, p][0]
                j = self.elastic_gradient_map[i, p][1]
                self.elastic_force[i, 0] += -self.elastic_gradient[e][0, j]
                self.elastic_force[i, 1] += -self.elastic_gradient[e][1, j]
                self.elastic_force[i, 2] += -self.elastic_gradient[e][2, j]

    @ti.func
    def ComputeEnergy(self, position, time_step) -> float:
        # sum_ = 0.0
        # for i,j in ti.ndrange(self.vertices_num,3):
        #     sum_ += (position[i,j]-1)**2
        # return sum_
        vertices_num = self.vertices_num
        inv_h = 1 / time_step
        coefficient = inv_h * inv_h / 2
        # y = ti.Matrix([[0.0 for i in range(3)] for j in range(self.vertices_num)])
        for i in range(vertices_num):
            for d in range(3):
                self.y[i, d] = self.position[i][d] + self.velocity[i][d] * time_step + self.external_acceleration[i][d] * time_step * time_step
        #kinetic_gradient = ti.Matrix([[0.0 for i in range(3)] for j in range(self.vertices_num)])
        #delta = ti.Matrix([[0.0 for i in range(3)] for j in range(self.vertices_num)])
        for i in range(vertices_num):
            for dd in range(3):
                x_next_d = position[i,dd]
                y_d = self.y[i, dd]
                #print("x_next_d: ", x_next_d)
                #print("y_d: ", y_d)
                self.delta[i, dd] = x_next_d - y_d
        kinetic_energy = 0.0
        for i, j, k in ti.ndrange(vertices_num, vertices_num, 3):
            kinetic_energy += coefficient * (self.int_density_matrix[i, j] * self.delta[j, k] * self.delta[i, k])
        # p0 = ti.Vector([0.0 for i in range(self.vertices_num)])
        # for iii, jjj in ti.ndrange(self.vertices_num, self.vertices_num):
        #     p0[iii] += self.int_density_matrix[iii, jjj] * delta_0[jjj]
        # kinetic_energy += delta_0.dot(p0)
        # p1 = ti.Vector([0.0 for i in range(self.vertices_num)])
        # for iii, jjj in ti.ndrange(self.vertices_num, self.vertices_num):
        #     p1[iii] += self.int_density_matrix[iii, jjj] * delta_1[jjj]
        # kinetic_energy += delta_1.dot(p1)
        # p2 = ti.Vector([0.0 for i in range(self.vertices_num)])
        # for iii, jjj in ti.ndrange(self.vertices_num, self.vertices_num):
        #     p2[iii] += self.int_density_matrix[iii, jjj] * delta_2[jjj]
        elastic_energy = self.ComputeElasticEnergy(position)
        # print("kinetic_energy: ", kinetic_energy)
        # print("elastic_energy: ", elastic_energy)
        return kinetic_energy + elastic_energy
    
    @ti.func
    def ComputeEnergyGradient(self, position, time_step):
        # for i,j in ti.ndrange(self.vertices_num, 3):
        #     self.energy_gradient[i,j] = 2*(position[i,j]-1)
        # return
        for i, j in ti.ndrange(self.vertices_num, 3):
            self.kinetic_gradient[i, j] = 0
        vertices_num = self.vertices_num
        inv_h = 1 / time_step
        coefficient = inv_h * inv_h
        # y = ti.Matrix([[0.0 for i in range(3)] for j in range(self.vertices_num)])
        for i in range(vertices_num):
            for d in range(3):
                self.y[i, d] = self.position[i][d] + self.velocity[i][d] * time_step + self.external_acceleration[i][d] * time_step * time_step
        #kinetic_gradient = ti.Matrix([[0.0 for i in range(3)] for j in range(self.vertices_num)])
        #delta = ti.Matrix([[0.0 for i in range(3)] for j in range(self.vertices_num)])
        for i in range(vertices_num):
            for dd in range(3):
                x_next_d = position[i,dd]
                y_d = self.y[i, dd]
                #print("x_next_d: ", x_next_d)
                #print("y_d: ", y_d)
                self.delta[i, dd] = x_next_d - y_d
        for iii, jjj, kkk in ti.ndrange(vertices_num, vertices_num, 3):
            self.kinetic_gradient[iii, kkk] += coefficient * (self.int_density_matrix[iii, jjj] * self.delta[jjj, kkk])
        #kinetic_gradient = self.int_density_matrix[None] @ delta
        self.ComputeElasticForce(position)
        # print("kinetic_gradient: ", kinetic_gradient)
        # print("elastic_gradient: ", elastic_gradient)
        for i, j in ti.ndrange(self.vertices_num, 3):
            self.energy_gradient[i, j] = self.kinetic_gradient[i, j] - self.elastic_force[i, j]
        
    @ti.func
    def E(self, x_next):
        x_next_ti = ti.field(dtype=ti.f64, shape=(self.vertices_num, 3))
        for i in range(self.vertices_num * 3):
            q = i // 3
            r = i % 3
            x_next_ti[q, r] = x_next[i]
        energy = self.ComputeEnergy(x_next_ti, self.h[None])
        return float(energy)
    
    @ti.func
    def E_gradient(self, x_next):
        x_next_ti = ti.field(dtype=ti.f64, shape=(self.vertices_num, 3))
        for i in range(self.vertices_num * 3):
            q = i // 3
            r = i % 3
            x_next_ti[q, r] = x_next[i]
        g = self.ComputeEnergyGradient(x_next, self.h[None])
        g1 = g * self.free_vertex[None]
        g1_np = np.array([0 for i in range(3 * self.vertices_num)])
        for i in range(self.vertices_num):
            for d in range(3):
                g1_np[i * 3 + d] = g1[i, d]
        return g1_np
    
    # @ti.func
    # def Optimizer(self):
        
    #     # res = scipy.optimize.minimize(self.E, self.x0_np, method="L-BFGS-B", jac=self.E_gradient, options={ "ftol": ftol, "maxiter": max_iter, "iprint": -1 })
    #     #print("x0_np: ", self.x0_np[None][0,0])
    #     return self.minimizer_LBFGS()
    
    @ti.func
    def minimizer_Adam(self,lr:ti.f64):
        ftol = 1e-3
        maxiter = 20
        b1 = 0.9
        b2 = 0.99
        
        self.ComputeEnergyGradient(self.x0_np, self.h[None])
        copy_fields_2d(self.energy_gradient, self.m_field, self.vertices_num, 3)
        copy_fields_2d_square(self.energy_gradient, self.v_field, self.vertices_num, 3)
        
        
        
        
        counter = 0
        
        ti.loop_config(serialize=True)
        for _ in range(maxiter):
            if counter >= 1:
                current_energy = self.ComputeEnergy(self.x0_next_np, self.h[None])
                # print(f"Current counter is {counter}, current energy is {current_energy}")
                
                # print("Current diff:", compute_difference_norm_2d(self.x0_next_np, self.x0_np, self.vertices_num, 3))
                if compute_difference_norm_2d(self.x0_next_np, self.x0_np, self.vertices_num, 3) < ftol:
                    # print("! Uses", counter, "iterations to converge")
                    break
            if counter > maxiter - 1:
                # print("Current diff:", compute_difference_norm_2d(self.x0_next_np, self.x0_np, self.vertices_num, 3))
                # print("Fatal Warning: minimizer did not converge")
                break
            
            
            counter += 1
            copy_fields_2d(self.x0_next_np, self.x0_np, self.vertices_num, 3)
            old_E = self.ComputeEnergy(self.x0_np, self.h[None])
            self.ComputeEnergyGradient(self.x0_next_np, self.h[None])
            self.update_m_v_adam(b1,b2,lr)
            
            counter += 1               
    
    @ti.func
    def update_m_v_adam(self,b1:ti.f64,b2:ti.f64,lr:ti.f64):
        EPSILON = ti.f64(1e-30)
        for i,j in ti.ndrange(self.vertices_num,3):
            self.m_field[i,j] = b1 * self.m_field[i,j] + (1-b1) * self.energy_gradient[i,j]
            self.v_field[i,j] = b2 * self.v_field[i,j] + (1-b2) * self.energy_gradient[i,j] * self.energy_gradient[i,j]
            self.x0_next_np[i,j] -= lr * self.m_field[i,j] / (self.v_field[i,j]**0.5 + EPSILON)
    
    @ti.func
    def minimizer_LBFGS(self):
        maxiter = 10
        ftol = 1e-15
        history_size = HISTORY_SIZE
        EPSILON = 1e-30
        counter = 0
        ti.loop_config(serialize=True)
        while True:
            if counter >= 1:
                current_energy = self.ComputeEnergy(self.x0_next_np, self.h[None])
                # print(f"Current counter is {counter}, current energy is {current_energy}")
                
                # print("Current diff:", compute_difference_norm_2d(self.x0_next_np, self.x0_np, self.vertices_num, 3))
                if compute_difference_norm_2d(self.x0_next_np, self.x0_np, self.vertices_num, 3) < ftol:
                    # print("! Uses", counter, "iterations to converge")
                    break
            if counter > maxiter - 1:
                # print("Current diff:", compute_difference_norm_2d(self.x0_next_np, self.x0_np, self.vertices_num, 3))
                # print("Fatal Warning: minimizer did not converge")
                break
            
            copy_fields_2d(self.x0_next_np, self.x0_np, self.vertices_num, 3)
            self.ComputeEnergyGradient(self.x0_next_np, self.h[None])
            copy_fields_2d(self.energy_gradient, self.g_field, self.vertices_num, 3)
            if counter > 0:
                copy_fields_2d(self.energy_gradient, self.q_field, self.vertices_num, 3)
                if counter > history_size:
                    self.obtain_q_when_large(counter, history_size)
                else:
                    # print("DDDDDDDDD")
                    self.obtain_q_when_small(counter,history_size)
                # print("CCCCCCCCCCCCbailurfjnwfkjdsals")
                compute_normed_2d(self.q_field,self.q_field,self.vertices_num,3)
                copy_fields_2d(self.q_field,self.step_direction, self.vertices_num, 3)
            else:
                # print("BBBBBBBBBBBBBBBBBdsuidafeoajwijnrdsioi")
                compute_normed_2d(self.g_field, self.step_direction, self.vertices_num,3)
                # copy_fields_2d(self.g_field,self.step_direction, self.vertices_num, 3)
            reverse_fields_2d(self.step_direction, self.vertices_num, 3)
            
            # print("Current step_direction[0,0]", self.step_direction[0,0])
            # print("Current step_direction[0,1]", self.step_direction[0,1])
            # print("Current step_direction[0,2]", self.step_direction[0,2])
            # print("Current step_direction[1,0]", self.step_direction[1,0])
            # print("Current step_direction[1,1]", self.step_direction[1,1])
            # print("Current step_direction[1,2]", self.step_direction[1,2])
            # print("Current step energy gradient[0,0]", self.energy_gradient[0,0])
            # # print("Current step energy == ", self.)
            # print("Current self.grad_history[0,0,0]==",self.grad_history[0,0,0])
            
            self.line_search()
            
            compute_add_with_mult_2d(self.x0_np, self.step_direction, self.alpha_chosen[None], self.x0_next_np, self.vertices_num, 3)
            
            
            
            self.ComputeEnergyGradient(self.x0_next_np, self.h[None])
            compute_add_with_mult_2d(self.energy_gradient, self.g_field, -1.0, self.y_temp, self.vertices_num, 3)
            
            self.update_history(counter,history_size,self.alpha_chosen[None])
            
            counter += 1
        
            
            
            
    @ti.func
    def update_history(self,counter:ti.i32,history_size:ti.i32,alpha:ti.f64):
        for vertex_idx, dim_idx in ti.ndrange(self.vertices_num,3):
            self.step_history[counter%history_size,vertex_idx,dim_idx] = alpha * self.step_direction[vertex_idx,dim_idx]
            self.grad_history[counter%history_size,vertex_idx,dim_idx] = self.y_temp[vertex_idx,dim_idx]
    
    @ti.func
    def line_search(self):
        current_min = float('inf')
        EPSILON = 1e-20
        current_index = -1
        best = 0
        ti.loop_config(serialize=True)
        for idx in range(N):
            current_alpha = alpha_field[idx]
            compute_add_with_mult_2d(self.x0_np, self.step_direction, current_alpha, self.x0_np_added, self.vertices_num, 3)
            if compute_difference_norm_2d(self.x0_np,self.x0_next_np,self.n,3) < self.n: # bound norm(delta) to self.n
                current_E = self.ComputeEnergy(self.x0_np_added, self.h[None])
                if current_E <= current_min:
                    current_index *= 0
                    current_index += idx
                    current_min = current_E
        
        # print("AAAAAAAAAAAAAbalsuioefljawlieuos")
        compute_normed_2d(self.step_direction,self.step_direction_normed,self.vertices_num,3)
        
        ti.loop_config(serialize=True)
        for idx in range(N):
            current_alpha = abs_alpha_field[idx]
            compute_add_with_mult_2d(self.x0_np, self.step_direction_normed, current_alpha, self.x0_np_added, self.vertices_num, 3)
            current_E = self.ComputeEnergy(self.x0_np_added, self.h[None])
            if current_E <= current_min:
                current_index *= 0
                current_index += idx
                current_min = current_E
                best += 1
        if current_index == -1:
            print("Oh No! Now current idx is -1, which is not right at all!")
            current_index += 1
        if best >= 1:
            self.alpha_chosen[None] = abs_alpha_field[current_index]/(norm_2d(self.step_direction,self.vertices_num,3)+EPSILON)
        else:
            self.alpha_chosen[None] = alpha_field[current_index]
        
    # @ti.func
    # def obtain_q_when_large(self,counter:ti.i32,history_size:ti.i32):

    #     EPSILON = 1e-25
    #     for index in range(history_size):
    #         self.dot_y_q[index] = 0.0
    #         self.dot_y_s[index] = 0.0
    #     ti.loop_config(serialize=True)
    #     for index in range(history_size):
    #         self.dot_y_q_index += 
    #     for index, vertex_index, dim_index in ti.ndrange(history_size,self.vertices_num, 3):
    #         self.dot_y_q[index] += self.grad_history[(-index+counter)%history_size,vertex_index,dim_index] * self.energy_gradient[vertex_index,dim_index]
    #         self.dot_y_s[index] += self.step_history[(-index+counter)%history_size,vertex_index,dim_index]*self.grad_history[(-index+counter)%history_size,vertex_index,dim_index]
    #     for index,vertex_index, dim_index in ti.ndrange(history_size,self.vertices_num, 3):
    #         self.q_field[vertex_index,dim_index] -= self.dot_y_q[index] * self.grad_history[(-index+counter)%history_size,vertex_index,dim_index] / (self.dot_y_s[index]+EPSILON)
        
    #     alpha = self.dot_y_q[(counter+1)%history_size]/(self.dot_y_s[(counter+1)%history_size]+EPSILON)
    #     for index in range(history_size):
    #         self.dot_y_q[index] = 0.0
    #         self.dot_y_s[index] = 0.0
    #     for index, vertex_index, dim_index in ti.ndrange(history_size,self.vertices_num, 3):
    #         self.dot_y_q[index] += self.grad_history[(counter+index)%history_size,vertex_index,dim_index] * self.energy_gradient[vertex_index,dim_index]
    #         self.dot_y_s[index] += self.step_history[(counter+index)%history_size,vertex_index,dim_index]*self.grad_history[(index+counter)%history_size,vertex_index,dim_index]
    #     for index, vertex_index, dim_index in ti.ndrange(history_size,self.vertices_num, 3):
    #         self.q_field[vertex_index,dim_index] += (alpha - self.dot_y_q[index]  / (self.dot_y_s[index]+EPSILON) ) * self.grad_history[(index+counter)%history_size,vertex_index,dim_index]
        
    @ti.func
    def obtain_q_when_large(self,counter:ti.i32,history_size:ti.i32):
        EPSILON = 1e-25
        ti.loop_config(serialize=True)
        for index in range(history_size):
            dot_y_q = dot_2d_and_3d(self.q_field,self.grad_history,(-index+counter)%history_size,self.vertices_num,3)
            dot_y_s = sub_dot(self.grad_history,self.step_history,(-index+counter)%history_size,self.vertices_num,3)
            alpha = dot_y_q / (dot_y_s + EPSILON)
            compute_add_with_mult_2d_add_3d(self.q_field,self.grad_history, -alpha, self.q_field, self.vertices_num,3,(-index+counter)%history_size)
        
        dot_y_q = dot_2d_and_3d(self.q_field,self.grad_history,(counter+1)%history_size,self.vertices_num,3)
        dot_y_s = sub_dot(self.grad_history,self.step_history,(counter+1)%history_size,self.vertices_num,3)
        alpha = dot_y_q / (dot_y_s + EPSILON)
        
        ti.loop_config(serialize=True)
        for index in range(history_size):
            dot_y_q = dot_2d_and_3d(self.q_field,self.grad_history,(index+counter)%history_size,self.vertices_num,3)
            dot_y_s = sub_dot(self.grad_history,self.step_history,(index+counter)%history_size,self.vertices_num,3)
            beta = dot_y_q / (dot_y_s + EPSILON)
            compute_add_with_mult_2d_add_3d(self.q_field,self.step_history, alpha-beta, self.q_field, self.vertices_num,3,(index+counter)%history_size)
            
    @ti.func
    def obtain_q_when_small(self,counter:ti.i32,history_size:ti.i32):
        EPSILON = 1e-15
        ti.loop_config(serialize=True)
        for index in range(counter):
            dot_y_q = dot_2d_and_3d(self.q_field,self.grad_history,(-index+counter-1)%history_size,self.vertices_num,3)
            dot_y_s = sub_dot(self.grad_history,self.step_history,(-index+counter-1)%history_size,self.vertices_num,3)
            alpha = dot_y_q / (dot_y_s + EPSILON)
            compute_add_with_mult_2d_add_3d(self.q_field, self.grad_history, -alpha, self.q_field, self.vertices_num,3,(-index+counter-1)%history_size)
        
        dot_y_q = dot_2d_and_3d(self.q_field,self.grad_history,(0)%history_size,self.vertices_num,3)
        dot_y_s = sub_dot(self.grad_history,self.step_history,(0)%history_size,self.vertices_num,3)
        alpha = dot_y_q / (dot_y_s + EPSILON)
        
        ti.loop_config(serialize=True)
        for index in range(counter):
            dot_y_q = dot_2d_and_3d(self.q_field,self.grad_history,(index)%history_size,self.vertices_num,3)
            dot_y_s = sub_dot(self.grad_history,self.step_history,(index)%history_size,self.vertices_num,3)
            beta = dot_y_q / (dot_y_s + EPSILON)
            compute_add_with_mult_2d_add_3d(self.q_field,self.step_history, alpha-beta, self.q_field, self.vertices_num,3,(index)%history_size)
            
    
    
    
    # @ti.func
    # def obtain_q_when_small(self,counter:ti.i32,history_size:ti.i32):
    #     EPSILON = 1e-25
    #     for index in range(history_size):
    #         self.dot_y_q[index] = 0.0
    #         self.dot_y_s[index] = 0.0
    #     for index,vertex_index, dim_index in ti.ndrange(counter, self.vertices_num, 3):
    #         self.dot_y_q[index] += self.grad_history[(counter - index - 1)%history_size,vertex_index,dim_index] * self.energy_gradient[vertex_index,dim_index]
    #         self.dot_y_s[index] += self.step_history[(counter - index - 1)%history_size,vertex_index,dim_index]*self.grad_history[(-index+counter)%history_size,vertex_index,dim_index]
    #     for index, vertex_index, dim_index in ti.ndrange(history_size,self.vertices_num, 3):
    #         self.q_field[vertex_index,dim_index] -= self.dot_y_q[index] * self.grad_history[(-index+counter-1)%history_size,vertex_index,dim_index] / (self.dot_y_s[index]+EPSILON)
        
    #     alpha = self.dot_y_q[(0)%history_size]/(self.dot_y_s[(0)%history_size]+EPSILON)
    #     for index in range(history_size):
    #         self.dot_y_q[index] = 0.0
    #         self.dot_y_s[index] = 0.0
    #     for index, vertex_index, dim_index in ti.ndrange(history_size,self.vertices_num, 3):
    #         self.dot_y_q[index] += self.grad_history[(index)%history_size,vertex_index,dim_index] * self.energy_gradient[vertex_index,dim_index]
    #         self.dot_y_s[index] += self.step_history[(index)%history_size,vertex_index,dim_index]*self.grad_history[(-index+counter)%history_size,vertex_index,dim_index]
    #     for index,vertex_index, dim_index in ti.ndrange(history_size,self.vertices_num, 3):
    #         self.q_field[vertex_index,dim_index] += (alpha - self.dot_y_q[index]  / (self.dot_y_s[index]+EPSILON) ) * self.grad_history[(index)%history_size,vertex_index,dim_index]
        
        
        
        
        
        
        
    # @ti.func
    # def minimizer(self):
    #     #print("init[0][0]: ", self.x0_np[None][0,0])
    #     options = dict(
    #         ftol = 1e-5,
    #         maxiter = 20000,
    #     )
    #     ftol = options["ftol"]
    #     maxiter = options["maxiter"]
    #     # assert method in ["Adam"]
    #     b1 = 0.9
    #     b2 = 0.99
    #     lr = 1e-3
        
    #     self.ComputeEnergyGradient(self.x0_np,self.h[None])
    #     copy_fields(self.energy_gradient, self.temp_m_field)
    #     copy_fields_square(self.energy_gradient, self.temp_v_field)
    #     old_pos = self.x0_np
    #     new_pos = self.x0_np
    #     success = False
    #     ti.loop_config(serialize=True)
    #     for _ in range(maxiter):
    #         old_E = self.ComputeEnergy(old_pos, self.h[None]) 
    #         # changed here
    #         self.ComputeEnergyGradient(old_pos,self.h[None])
    #         m += -m + b1 * m + (1-b1) * self.energy_gradient
    #         v += -v + b2 * v + (1-b2) * self.energy_gradient
    #         ##
    #         new_pos -= lr*m / (v**0.5 + 1e-13)
    #         new_E = self.ComputeEnergy(new_pos, self.h[None]) 
    #         old_pos -= lr*m / (v**0.5 + 1e-13)
    #         if abs(new_E-old_E)/abs(old_E) < ftol:
    #             success = True
    #             break
    #         if abs(new_E-old_E) < ftol:
    #             success = True
    #             break
    #     if not success:
    #         print("Failed to converge")
    #     return new_pos

    
    
    @ti.kernel
    def Forward(self, time_step: ti.f64):
        #print(self.position[0])
        for i, d in ti.ndrange(self.vertices_num, 3):
            self.x0_np[i,d] = self.position[i][d]
            self.x0_next_np[i,d] = self.next_position[i][d]
        # print(self.position[0])
        # print(self.x0_np[None])
        # print(self.velocity[0])
        # print(self.velocity[1])
        # print(self.velocity[2])
        # print(self.velocity[3])
        self.kernel_Forward(time_step)
        copy_fields(self.next_position, self.position)
        copy_fields(self.next_velocity, self.velocity)
    @ti.func
    def Optimize(self):
        copy_fields_2d(self.x0_np, self.x0_next_np,self.n,3)
        self.minimizer_Adam(1e-5)
        self.minimizer_Adam(1e-6)
        self.minimizer_Adam(1e-7)
        self.minimizer_LBFGS()
    @ti.func
    def kernel_Forward(self, time_step: ti.f64):
        # print(self.position)
        self.h[None] = time_step  
        
          
        # Optimize
        #print("x0_np: ", self.x0_np[None][0,0])
        self.Optimize()
        inv_h = 1 / self.h[None]
        for i in range(self.vertices_num):
            for d in range(3):
                if self.free_vertex_vector_field[i][d] == 1:
                    self.next_position[i][d] = self.x0_next_np[i, d]
                    self.next_velocity[i][d] = (self.next_position[i][d] - self.position[i][d]) * inv_h
                else:
                    self.next_position[i][d] = self.dirichlet_boundary_condition[i][d]
                    self.next_velocity[i][d] = 0.0