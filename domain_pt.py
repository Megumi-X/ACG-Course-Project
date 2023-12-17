import torch
torch.set_default_dtype(torch.float64)

class Domain(torch.nn.Module):
    def __init__(self,vertices_num:int,elements_num:int):
        super().__init__()
        self.vertices_num = vertices_num
        self.elements_num = elements_num
        self.vertices = torch.nn.Parameter(torch.zeros((vertices_num,3),dtype=torch.float64),requires_grad=False)
        self.elements = torch.nn.Parameter(torch.zeros((elements_num,4),dtype=torch.long),requires_grad=False)
        self.finite_elements_vertices = torch.nn.Parameter(torch.zeros((elements_num,4,3),dtype=torch.float64),requires_grad=False)
        self.finite_elements_polynomials = torch.nn.Parameter(torch.zeros((elements_num,4,4),dtype=torch.float64),requires_grad=False)
        self.finite_elements_geometry_info_measure = torch.nn.Parameter(torch.zeros((elements_num,4,3),dtype=torch.float64),requires_grad=False)
    
    def kernel_Initialize(self,vertices,elements):
        self.vertices.data.copy_(vertices)
        # print(self.vertices)
        self.elements.data.copy_(elements)
        # self.finite_elements_vertices
        for i in range(self.elements_num):
            v0 = self.vertices[self.elements[i][0]]
            v1 = self.vertices[self.elements[i][1]]
            v2 = self.vertices[self.elements[i][2]]
            v3 = self.vertices[self.elements[i][3]]
            for ii in range(3):
                self.finite_elements_vertices[i,0,ii] = v0[ii]
                self.finite_elements_vertices[i,1,ii] = v1[ii]
                self.finite_elements_vertices[i,2,ii] = v2[ii]
                self.finite_elements_vertices[i,3,ii] = v3[ii]
            volume = (v1-v0).cross(v2-v1).dot(v3-v2) / 6.
            if volume < 0:
                order = torch.tensor([0,1,2,3])
                volume = - volume
            else:
                order = torch.tensor([0,2,1,3])
            
            self.finite_elements_geometry_info_measure[i,3,0] = volume
    
    def Initialize(self,vertices,elements):
        self.kernel_Initialize(vertices,elements)
        # self.finite_elements_vert
        A_inv = torch.zeros([self.elements_num,4,4],dtype=torch.float64)
        for i in range(self.elements_num):
            a = torch.zeros([4,4],dtype=torch.float64)
            for ii in range(4):
                a[0,ii] = self.finite_elements_vertices[i,ii,0]
                a[1,ii] = self.finite_elements_vertices[i,ii,1]
                a[2,ii] = self.finite_elements_vertices[i,ii,2]
                a[3,ii] = 1.0
            # print("biauwrhf8qwuberaieushrwek")
            # print(self.finite_elements_vertices[i])
            # print(a)
            # print(torch.det(a))
            A_inv[i] = torch.inverse(a)
            # print(A_inv[i])
        self.finite_elements_polynomials.data.copy_(A_inv)
            