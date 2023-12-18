import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm
from deformable_pt import DeformableSimulator, DeformableSimulatorController

torch.set_default_dtype(torch.float64)
USE_CUDA = True

def create_folder(folder_name, exist_ok):
    Path(folder_name).mkdir(parents=True, exist_ok=exist_ok)


USE_CUDA = False
X = 10
Y = 30
dx = 0.02
vertices_num = (X + 1) * (Y + 1) * 2
elements_num = X * Y * 6


init_vertices = torch.zeros([vertices_num,3],dtype=torch.float64)

for i in range(Y+1):
    for j in range(X+1):
        init_vertices[(i*(X+1)+j)*2] += torch.tensor([j*dx,i*dx,0.0])
        init_vertices[(i*(X+1)+j)*2 + 1] += torch.tensor([j*dx,i*dx,dx])

elements = torch.zeros([elements_num,4],dtype=torch.long)

for i in range(Y):
    for j in range(X):
        a = (i * (X + 1) + j) * 2
        b = a + 1
        c = a + 2
        d = a + 3
        e = a + 2 * (X + 1)
        f = e + 1
        g = e + 2
        h = e + 3
        elements[(i*X + j)*6]     = torch.tensor([a,c,d,h])
        elements[(i*X + j)*6 + 1] = torch.tensor([a,b,d,h])
        elements[(i*X + j)*6 + 2] = torch.tensor([a,b,f,h])
        elements[(i*X + j)*6 + 3] = torch.tensor([a,e,f,h])
        elements[(i*X + j)*6 + 4] = torch.tensor([a,e,g,h])
        elements[(i*X + j)*6 + 5] = torch.tensor([a,c,g,h])

print("Initializing...")
simulator = DeformableSimulator(vertices_num, elements_num)
dirichilet_boundary = torch.zeros([vertices_num,3],dtype=torch.float64)
dirichilet_boundary += float('inf')
for j in range(X + 1):
    dirichilet_boundary[2 * j] = init_vertices[2 * j]
    dirichilet_boundary[2 * j + 1] = init_vertices[2 * j + 1]
    


simulator.Initialize(init_vertices, elements, 1e3, 1e7, 0.3, dirichilet_boundary)
# simulator.Initialize(init_vertices, elements, 1e3, 1e6, 0.3)
print("Initialization finished.")


for index in range(simulator.vertices_num):
    simulator.external_acceleration[index] += torch.tensor([0.0, 0.0, -9.80])

# for j in range(X+1):
#     simulator.dirichlet_boundary_condition[j * 2] = init_vertices[j * 2]
#     simulator.dirichlet_boundary_condition[j * 2 + 1] = init_vertices[j * 2 + 1]


element_np = simulator.undeformed.elements.numpy()
folder = Path("./") / "simple_cloth"
create_folder(folder, exist_ok=True)
np.save(folder / "elements.npy", element_np)
position_0 = simulator.position.numpy()
np.save(folder / "0000.npy", position_0)

#print(simulator.external_acceleration.numpy())


simulatorController = DeformableSimulatorController(simulator)

if USE_CUDA:
    simulatorController.cuda()

# for f in tqdm(range(100)):
#     position_np = simulator.position.detach().cpu().numpy()
#     print("Current step is {} and current position - 0.01 is {}".format(f,position_np[:,2].mean() - 0.01))
for f in tqdm(range(300)):
    # position_np = simulator.position.numpy()
    #print("Current step is {} and current position - 0.01 is {}".format(f,position_np[:,2].mean() - 0.01))
    # set_force(ti.cos(f * 0.1) * 5)
    simulatorController.Forward(0.01)
    position_np = simulator.position.detach().cpu().numpy()
    np.save(folder / "{:04d}.npy".format(f + 1), position_np)
