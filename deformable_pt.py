import torch
from domain_pt import Domain
from no_break_optimizer import NoBreak_LBFGS

torch.set_grad_enabled(True)

MAGIC = 999999999

def DeterminantGrad(A):
    dJdA = torch.zeros_like(A)
    dJdA[...,0, 0] = A[...,1, 1] * A[...,2, 2] - A[...,1, 2] * A[...,2, 1]
    dJdA[...,0, 1] = A[...,2, 0] * A[...,1, 2] - A[...,1, 0] * A[...,2, 2]
    dJdA[...,0, 2] = A[...,1, 0] * A[...,2, 1] - A[...,2, 0] * A[...,1, 1]
    dJdA[...,1, 0] = A[...,2, 1] * A[...,0, 2] - A[...,0, 1] * A[...,2, 2]
    dJdA[...,1, 1] = A[...,0, 0] * A[...,2, 2] - A[...,0, 2] * A[...,2, 0]
    dJdA[...,1, 2] = A[...,2, 0] * A[...,0, 1] - A[...,0, 0] * A[...,2, 1]
    dJdA[...,2, 0] = A[...,0, 1] * A[...,1, 2] - A[...,1, 1] * A[...,0, 2]
    dJdA[...,2, 1] = A[...,1, 0] * A[...,0, 2] - A[...,0, 0] * A[...,1, 2]
    dJdA[...,2, 2] = A[...,0, 0] * A[...,1, 1] - A[...,0, 1] * A[...,1, 0]
    return dJdA

def ComputeEnergyDensity(F,lam,mu):
    '''
        Input:
            F: ... * 3 * 3
            lam: ...
            mu: ...
        Output:
            ...
    '''
    EPSILON = 1e-30
    C = torch.einsum('...ji,...jk->...ik',F,F) # ... * 3 * 3
    J = torch.linalg.det(F) # ...*3*3 -> ...
    Ic = C[...,0,0] + C[...,1,1] + C[...,2,2] # ...
    delta = 1.
    dim = 3
    alpha = (1-1/(dim+delta)) * mu/lam + 1
    Ic_verified = torch.nn.functional.relu(Ic + delta, inplace=False)
    return mu/2*(Ic-dim) + lam/2 * (J-alpha)**2 - 0.5 * mu * torch.log(Ic_verified+EPSILON)

def ComputeStressDensity(F,lam,mu):
    '''
        Input:
            F: ... * 3 * 3
            lam: ...
            mu: ...
        Output:
            ... * 3 * 3
    '''
    C = torch.einsum('...ji,...jk->...ik',F,F) # ... * 3 * 3
    J = torch.linalg.det(F) # ...*3*3 -> ...
    Ic = C[...,0,0] + C[...,1,1] + C[...,2,2] # ...
    delta = 1.
    dim = 3
    alpha = (1-1/(dim+delta)) * mu/lam + 1
    dJdF = DeterminantGrad(F)
    return (1 - 1 / (Ic+delta)) * mu * F + lam * (J - alpha) * dJdF
    




class DeformableSimulator(torch.nn.Module):
    def __init__(self, vertices_num,element_num):
        super().__init__()
        
        self.next_position = torch.nn.Parameter(torch.zeros((vertices_num,3),dtype=torch.float64), requires_grad=True)
        self.undeformed = Domain(vertices_num, element_num)
        self.material_density = torch.nn.Parameter(torch.zeros((element_num),dtype=torch.float64), requires_grad=False)
        self.material_youngs_modulus = torch.nn.Parameter(torch.zeros((element_num),dtype=torch.float64), requires_grad=False)
        self.material_poissons_ratio = torch.nn.Parameter(torch.zeros((element_num),dtype=torch.float64), requires_grad=False)
        self.material_lam = torch.nn.Parameter(torch.zeros((element_num),dtype=torch.float64), requires_grad=False)
        self.material_mu = torch.nn.Parameter(torch.zeros((element_num),dtype=torch.float64), requires_grad=False)
        
        self.external_acceleration = torch.nn.Parameter(torch.zeros((vertices_num,3),dtype=torch.float64),requires_grad=False)
        self.dirichlet_boundary_condition = torch.nn.Parameter(torch.zeros((vertices_num,3),dtype=torch.float64), requires_grad=False)
        self.dirichlet_value = torch.nn.Parameter(torch.zeros((vertices_num,3),dtype=torch.float64), requires_grad=False)
        
        self.int_matrix = torch.nn.Parameter(torch.zeros((1000,1000),dtype=torch.float64), requires_grad=False)
        self.int_density_matrix = torch.nn.Parameter(torch.zeros((vertices_num,vertices_num),dtype=torch.float64), requires_grad = False)
        
        
        self.free_vertex_vector_field = torch.nn.Parameter(torch.zeros([vertices_num,3],dtype=torch.long),requires_grad=False)
        self.elastic_gradient_map = torch.nn.Parameter(torch.zeros((vertices_num,100,2),dtype=torch.float64), requires_grad=False)
        
        self.vertices_num = vertices_num
        self.element_num = element_num
        self.free_vertex = torch.nn.Parameter(torch.zeros((vertices_num,3),dtype=torch.long), requires_grad=False)
        self.h = torch.nn.Parameter(torch.zeros((),dtype=torch.float64), requires_grad=False)
        self.n = vertices_num
        
        
        self.position = torch.nn.Parameter(torch.zeros((vertices_num,3),dtype=torch.float64), requires_grad=False)
        
        self.velocity = torch.nn.Parameter(torch.zeros((vertices_num,3),dtype=torch.float64), requires_grad=False)
        
        
        
        
        
        
    def InitializePosition(self):
        vertices_num = self.undeformed.vertices_num
        with torch.no_grad():
            self.position[:] = self.undeformed.vertices[:]
            self.next_position[:] = self.undeformed.vertices[:]
            for i in range(vertices_num):
                for dim in range(3):
                    self.dirichlet_boundary_condition[i,dim] = float('inf')
    
    def ComputeIntMatrix(self):
        elements_num = self.undeformed.elements_num
        self.int_density_matrix *= 0
        self.int_density_matrix *= 0
        for e in range(elements_num):
            finite_element = self.undeformed.elements[e]
            element = self.undeformed.elements[e]
            local_matrix = torch.zeros((4,4),dtype=torch.float64)
            for i in range(4):
                phi_i = torch.tensor([self.undeformed.finite_elements_polynomials[e][i][j] for j in range(4)])
                for j in range(i,4):
                    phi_j = torch.tensor([self.undeformed.finite_elements_polynomials[e][j][k] for k in range(4)])
                    pos = torch.zeros([3],dtype=torch.float64)
                    pos += self.undeformed.finite_elements_vertices[e].sum(dim=0)
                    pos /= 4
                    
                    value = (torch.dot(phi_i[:3],pos) + phi_i[3])*(torch.dot(phi_j[:3],pos) + phi_j[3])
                    
                    w_ij = value * self.undeformed.finite_elements_geometry_info_measure[e,3,0]
                    
                    if i == j:
                        local_matrix[i,j] += w_ij
                        
                        self.int_matrix[int(element[i]),int(element[i])] += w_ij
                        
                        self.int_density_matrix[int(element[i]),int(element[i])] += w_ij * self.material_density[e]
                    else:
                        local_matrix[i,j] = w_ij
                        local_matrix[j,i] = w_ij
                        
                        self.int_matrix[element[i],element[j]] += w_ij
                        self.int_matrix[element[j],element[i]] += w_ij
                        
                        self.int_density_matrix[element[i],element[j]] += w_ij * self.material_density[e]
                        self.int_density_matrix[element[j],element[i]] += w_ij * self.material_density[e]
                
    def Initialize(self,vertices,elements,density,youngs_modulus, poissons_ratio, dirichlet_boundary_condition = None):
        self.undeformed.Initialize(vertices, elements)
        
        elements_num = self.undeformed.elements_num
        
        for i in range(elements_num):
            self.material_density[i] = density
            self.material_youngs_modulus[i] = youngs_modulus
            self.material_poissons_ratio[i] = poissons_ratio
            self.material_lam[i] = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
            self.material_mu[i] = youngs_modulus / (2 * (1 + poissons_ratio))
        
        self.InitializePosition()
        
        if dirichlet_boundary_condition is not None:
            self.dirichlet_boundary_condition.data.copy_(dirichlet_boundary_condition)
        
        self.ComputeIntMatrix()
        
        for e in range(self.vertices_num):
            for i in range(100):
                for j in range(2):
                    self.elastic_gradient_map[e,i,j] = MAGIC
        
        for e in range(elements_num):
            element = self.undeformed.elements[e]
            for i in range(4):
                num = 0
                for j in range(100):
                    if self.elastic_gradient_map[element[i],j,0] == MAGIC:
                        num = j
                        break
                self.elastic_gradient_map[element[i],j,0] = e
                self.elastic_gradient_map[element[i],j,1] = i
        for i in range(self.vertices_num):
            for d in range(3):
                if self.dirichlet_boundary_condition[i,d] == float('inf'):
                    self.free_vertex[i,d] = 1
                    self.free_vertex_vector_field[i,d] = 1
                else:
                    self.free_vertex[i,d] = 0
                    self.free_vertex_vector_field[i,d] = 0
                    self.dirichlet_value[i,d] = self.dirichlet_boundary_condition[i,d]
    
    def ComputeElasticEnergy(self,position):
        basis_derivatives_q = self.undeformed.finite_elements_polynomials[:,:4,:3]
        fin_ele = self.undeformed.elements
        
        row_indices = torch.arange(fin_ele.size(0)).view(-1, 1)
        row_indices = row_indices.expand(fin_ele.size(0), fin_ele.size(1))
        # Use advanced indexing to get the desired values
        local_position = position[fin_ele[row_indices, torch.arange(fin_ele.size(1))], :]
        
        # print("local_pos.shape==",local_position.shape)
        # print("basis_....shape==,",basis_derivatives_q.shape)
        F = torch.einsum('bft,bfl->btl',local_position, basis_derivatives_q)
        
        
        return torch.einsum('b,b->',ComputeEnergyDensity(F,self.material_lam,self.material_mu) , self.undeformed.finite_elements_geometry_info_measure[:,3,0])
    
    # def ComputeElasticForce(self,position):
    #     element_num = self.undeformed.elements_num
    #     basis_derivatives_q = self.undeformed.finite_elements_polynomials[:,:4,:3]
    #     fin_ele = self.undeformed.elements
        
    #     row_indices = torch.arange(fin_ele.size(0)).view(-1, 1)
    #     row_indices = row_indices.expand(fin_ele.size(0), fin_ele.size(1))
    #     # Use advanced indexing to get the desired values
    #     local_position = position[fin_ele[row_indices, torch.arange(fin_ele.size(1))], :]
        
    #     F = torch.einsum('btf,bfl->btl',local_position, basis_derivatives_q)
    #     P = ComputeStressDensity(F,self.material_lam,self.material_mu)
        
    #     elastic_gradient = 
    
    def ComputeEnergy(self,position,time_step):
        # return (position**2).sum()
        # vertices_num = self.vertices_num
        inv_h = 1/time_step
        coefficient = inv_h**2/2
        # print(self.external_acceleration)
        # print("pos.device==",self.position.device)
        # print("velocity.device==",self.velocity.device)
        # print("external_acceleration.device==",self.external_acceleration.device)
        y = self.position + self.velocity * time_step + self.external_acceleration * time_step**2
        
        delta = position - y # vert * 3
        
        delta_delta = torch.einsum('...jk,...ik->...ji',delta,delta)
        kinetic_energy = torch.einsum('...ij,...ji->',self.int_density_matrix,delta_delta) * coefficient
        elastic_energy = self.ComputeElasticEnergy(position)
        

        return dict(
            energy = kinetic_energy + elastic_energy,
            kinetic_energy = kinetic_energy,
            elastic_energy = elastic_energy,
        )
    def loss_function(self,time_step):
        return self.ComputeEnergy(self.next_position,time_step)
    def Forward(self,time_step):
        loss = self.loss_function(time_step)
        return loss

    def UpdatePositionAndVelocity(self, time_step):
        inv_h = 1/time_step        
        with torch.no_grad():
            self.next_position[self.free_vertex_vector_field==0] *= 0
            self.next_position[self.free_vertex_vector_field==0] += self.dirichlet_boundary_condition[self.free_vertex_vector_field==0]
            self.velocity[self.free_vertex_vector_field==0] *= 0.0     
            self.velocity[self.free_vertex_vector_field==1] *= 0.0     
            self.velocity[self.free_vertex_vector_field==1] += inv_h * (self.next_position - self.position)[self.free_vertex_vector_field==1]

        with torch.no_grad():
            self.position *= 0
            self.position += self.next_position
        
    def UpdateNextPosition(self):
        with torch.no_grad():
            self.next_position *= 0
            self.next_position += self.position
                
    
        
    def forward(self,*args,**kwargs):
        return self.Forward(*args,**kwargs)


class DeformableSimulatorController(torch.nn.Module):
    def __init__(self,model):
        super().__init__()
        self.model = model

        self.Adam_MaxIter = 100
        self.optimizer_Adam_list = [
        #     torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),lr=1e-6),
        #     torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),lr=1e-7),
        #     torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),lr=1e-8),
        #     torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),lr=1e-9),
        ]    
        
        
        self.optimizer_LBFGS_list = [
            
            NoBreak_LBFGS(filter(lambda p: p.requires_grad, self.model.parameters()),max_iter=1500,lr=1e-0,line_search_fn='strong_wolfe',
                                           tolerance_grad = 1e-15, tolerance_change = 1e-15),
            # torch.optim.LBFGS(filter(lambda p: p.requires_grad, self.model.parameters()),max_iter=1500,lr=1e-1,line_search_fn='strong_wolfe',
            #                                tolerance_grad = 1e-15, tolerance_change = 1e-15),
            # torch.optim.LBFGS(filter(lambda p: p.requires_grad, self.model.parameters()),max_iter=500,lr=1e-2,line_search_fn='strong_wolfe',
            #                                tolerance_grad = 1e-15, tolerance_change = 1e-15),
            NoBreak_LBFGS(filter(lambda p: p.requires_grad, self.model.parameters()),max_iter=1500,lr=1e-3,line_search_fn='strong_wolfe',
                                           tolerance_grad = 1e-15, tolerance_change = 1e-15),
            # # torch.optim.LBFGS(filter(lambda p: p.requires_grad, self.model.parameters()),max_iter=500,lr=1e-3,line_search_fn='strong_wolfe',
            # #                                tolerance_grad = 1e-15, tolerance_change = 1e-15),
            NoBreak_LBFGS(filter(lambda p: p.requires_grad, self.model.parameters()),max_iter=1500,lr=1e-7,line_search_fn='strong_wolfe',
                                           tolerance_grad = 1e-15, tolerance_change = 1e-15),
        ]
        
        
    def Forward(self,time_step):
        self.model.UpdateNextPosition()
        for optimizer in self.optimizer_Adam_list:
            for _ in range(self.Adam_MaxIter):
                optimizer.zero_grad()
                loss = self.model.Forward(time_step)
                # print("Current energy is ", loss)
                loss['energy'].backward()
                optimizer.step()
            
        for optimizer in self.optimizer_LBFGS_list:
                def closure():
                    optimizer.zero_grad()
                    loss = self.model.Forward(time_step)
                    # print("Current energy is ", loss['energy'].item())
                    # print("Current Kinetic energy is ", loss['kinetic_energy'].item())
                    # print("Current Elastic energy is ", loss['elastic_energy'].item())
                    loss['energy'].backward()
                    return loss['energy']
                optimizer.step(closure)
        
        self.model.UpdatePositionAndVelocity(time_step)