import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm
from deformable_pt import DeformableSimulator, DeformableSimulatorController
from tet_mesh import tetrahedralize
from pbrt_renderer import to_real_array, to_integer_array

def file_exist(file_name):
    return Path(file_name).is_file()

def create_and_load_tet_meshes(folder, mesh_name):
    if not file_exist(folder / "{}_vertices.npy".format(mesh_name)) \
        or not file_exist(folder / "{}_elements.npy".format(mesh_name)):
        # Create the tet mesh.
        vertices, elements = tetrahedralize(Path("asset") / "{}.obj".format(mesh_name),
            visualize=False, normalize_input=False, options={
            "switches": "pq1.1/0Ya2e-5V"
        })
        np.save(folder / "{}_vertices".format(mesh_name), vertices)
        np.save(folder / "{}_elements".format(mesh_name), elements)

    vertices = np.load(folder / "{}_vertices.npy".format(mesh_name))
    elements = np.load(folder / "{}_elements.npy".format(mesh_name))
    print("Loading {}...{} vertices, {} elements".format(mesh_name, vertices.shape[0], elements.shape[0]))
    return to_real_array(vertices), to_integer_array(elements)

def create_folder(folder_name, exist_ok):
    Path(folder_name).mkdir(parents=True, exist_ok=exist_ok)

folder = Path("./") / "ball_on_pin"
create_folder(folder, exist_ok=True)

vertices_np, elements_np = create_and_load_tet_meshes(folder, "sphere_low_res")
vertices_num = vertices_np.shape[0]
elements_num = elements_np.shape[0]

init_vertices = torch.tensor(vertices_np, dtype=torch.float64)
elements = torch.tensor(elements_np, dtype=torch.long)
init_vertices += torch.tensor([0, 0, 3])

print("Initializing...")
simulator = DeformableSimulator(vertices_num, elements_num)
simulator.Initialize(init_vertices, elements, 1e3, 1e4, 0.3)

def ground_collision(position):
    return position[:, 2]

def pin_collision(position):
    r = torch.norm(position[:, :2], dim=1)
    z = position[:, 2]
    return (2 * r + z - 1) / np.sqrt(5)



simulator.collision_bound.append(ground_collision)
simulator.collision_bound.append(pin_collision)
for i in range(vertices_num):
    simulator.external_acceleration[i] = torch.tensor([0, 0, -9.8])
print("Initialization finished.")

element_np = simulator.undeformed.elements.numpy()
np.save(folder / "elements.npy", element_np)
position_0 = simulator.position.numpy()
np.save(folder / "0000.npy", position_0)

simulatorController = DeformableSimulatorController(simulator)
simulatorController.cuda()

for f in tqdm(range(200)):
    simulatorController.Forward(0.01)
    position_np = simulator.position.detach().cpu().numpy()
    np.save(folder / "{:04d}.npy".format(f + 1), position_np)