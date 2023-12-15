import taichi as ti
from finite_element import FiniteElement
from finite_element import finiteElementTypeDict, GeometryShape, IntegratePoly
from finite_element import GeometryShape
import numpy as np
@ti.data_oriented
class Domain:
    def __init__(self, vertices_num: ti.int32, elements_num: ti.int32):
        self.vertices_num = vertices_num
        self.elements_num = elements_num
        self.vertices = ti.Vector.field(n=3, dtype=ti.f32, shape=self.vertices_num)
        self.elements = ti.Vector.field(n=4, dtype=ti.int32, shape=self.elements_num)
        self.finite_elements = ti.Struct.field(finiteElementTypeDict, shape=self.elements_num)
        self.geometry_info = ti.Struct.field({
            'dim': ti.i32,
            'vertices_num': ti.i32,
            'vertex_indices': ti.types.vector(4, ti.i32),
            'measure': ti.f32,
        }, shape=(4,6))

    def Initialize(self, vertices, elements):
        for iii in range(self.vertices_num):
            for jjj in range(3):
                self.vertices[iii][jjj] = vertices[iii][jjj]
        #for iii in range(self.elements_num):
            #for jjj in range(4):
        self.elements = elements
        for i in range(self.elements_num):
            v0 = self.vertices[self.elements[i][0]]
            print(v0)
            v1 = self.vertices[self.elements[i][1]]
            v2 = self.vertices[self.elements[i][2]]
            v3 = self.vertices[self.elements[i][3]]
            self.finite_elements[i].vertices_num = 4
            for ii in range(4):
                self.finite_elements[i].vertices[0,i] = v0[i]
                self.finite_elements[i].vertices[1,i] = v1[i]
                self.finite_elements[i].vertices[2,i] = v2[i]
                self.finite_elements[i].vertices[3,i] = v3[i]
            
            volume = np.dot(np.cross((v1 - v0),(v2 - v1)),(v3 - v2)) / 6
            order = ti.Vector([0,0,0,0])
            if volume < 0 :
                order += ti.Vector([0, 1, 2, 3])
                volume = - volume
            else:
                order += ti.Vector([0, 2, 1, 3])
            for ii in range(4):
                self.finite_elements[i].geometry_info_dim[0,ii] = 0
                self.finite_elements[i].geometry_info_vertices_num[0,ii] = 1
                self.finite_elements[i].geometry_info_vertex_indices_0[ii,0] = ii
                self.finite_elements[i].geometry_info_vertex_indices_0[ii,1] = ii
                self.finite_elements[i].geometry_info_vertex_indices_0[ii,2] = ii
                self.finite_elements[i].geometry_info_vertex_indices_0[ii,3] = ii
                self.finite_elements[i].geometry_info_measure[0,ii] = 0.0
            edge_index = 0
            for ii in range(4):
                for jj in range(ii + 1, 4):
                    self.finite_elements[i].geometry_info_dim[1,edge_index] = 1
                    self.finite_elements[i].geometry_info_vertices_num[1,edge_index] = 2
                    self.finite_elements[i].geometry_info_vertex_indices_1[edge_index,0] = ii
                    self.finite_elements[i].geometry_info_vertex_indices_1[edge_index,1] = jj
                    print(self.finite_elements[i].vertices)
                    M = self.finite_elements[i].vertices.m
                    self.finite_elements[i].geometry_info_measure[1,edge_index] = (ti.Vector([self.finite_elements[i].vertices[ii,jjj] for jjj in range(M)]) - ti.Vector([self.finite_elements[i].vertices[jj,jjj] for jjj in range(M)])).norm()
                    edge_index += 1
            for ii in range(6):
                for jj in range(4):
                    self.finite_elements[i].geometry_info_vertex_indices_2[ii,jj] = 0
                    
            self.finite_elements[i].geometry_info_vertex_indices_2 += ti.Matrix([[order[0], order[1], order[2],0],
                       [order[1], order[0], order[3],0],
                       [order[2], order[1], order[3],0],
                       [order[0], order[2], order[3],0],
                       [0,0,0,0],
                       [0,0,0,0]])
            # self.finite_elements[i].geometry_info_vertex_indices_2[0,:] += ti.Vector([order[0], order[1], order[2],0])
            # self.finite_elements[i].geometry_info_vertex_indices_2[1,:] += ti.Vector([order[1], order[0], order[3],0])
            # self.finite_elements[i].geometry_info_vertex_indices_2[2,:] += ti.Vector([order[2], order[1], order[3],0])
            # self.finite_elements[i].geometry_info_vertex_indices_2[3,:] += ti.Vector([order[0], order[2], order[3],0])
            
            
            for ii in range(4):
                self.finite_elements[i].geometry_info_dim[2,ii] = 2
                self.finite_elements[i].geometry_info_vertices_num[2,ii] = 3
                v00 = self.finite_elements[i].geometry_info_vertex_indices_2[ii,0]
                v01 = self.finite_elements[i].geometry_info_vertex_indices_2[ii,1]
                v02 = self.finite_elements[i].geometry_info_vertex_indices_2[ii,2]
                v001 = matrix_row_to_vec(self.finite_elements[i].vertices,v01) - matrix_row_to_vec(self.finite_elements[i].vertices,v00)
                v002 = matrix_row_to_vec(self.finite_elements[i].vertices,v02) - matrix_row_to_vec(self.finite_elements[i].vertices,v00)
                self.finite_elements[i].geometry_info_measure[2,i] = v001.cross(v002).norm() / 2
            
            for jjj in range(4):
                self.finite_elements[i].geometry_info_vertex_indices_3[0,jjj]= order[jjj]
            self.finite_elements[i].geometry_info_dim[3,0] = 3
            self.finite_elements[i].geometry_info_vertices_num[3,0] = 4
            self.finite_elements[i].geometry_info_measure[3,0] = volume
            
            A = ti.Matrix([[0.0 for ii in range(self.vertices_num)] for jj in range(4)])
            for ii in range(4):
                A[0, ii] = self.finite_elements[i].vertices[ii,0]
                A[1, ii] = self.finite_elements[i].vertices[ii,1]
                A[2, ii] = self.finite_elements[i].vertices[ii,2]
                A[3, ii] = 1.0
            
            A_inv = A.inverse()
            for i in range(4):
                self.finite_elements[i].polynomials[i,0] = A_inv[i, 0]
                self.finite_elements[i].polynomials[i,1] = A_inv[i, 1]
                self.finite_elements[i].polynomials[i,2] = A_inv[i, 2]
                self.finite_elements[i].polynomials[i,3] = A_inv[i, 3]
            
                    
                    
                    
                    
                    
                    
                    
            # self.finite_elements[i].Initialize(self.elements[i, 0], self.elements[i, 1], self.elements[i, 2], self.elements[i, 3])

        # Set geometry info
        # 0D
        for i in range(self.vertices_num):
            self.geometry_info[0, i].dim = 0
            self.geometry_info[0, i].vertices_num = 1
            self.geometry_info[0, i].vertex_indicies = i
            self.geometry_info[0, i].measure = 0.0
              

def matrix_row_to_vec(A: ti.template(), idx:ti.i32):
    vector_ = ti.Vector([A[idx,j] for j in range(A.m)])
    return vector_