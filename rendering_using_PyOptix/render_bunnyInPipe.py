import os
import matplotlib.pyplot as plt
import numpy as np
from plotoptix import NpOptiX
from plotoptix.utils import make_color, map_to_colors, make_color_2d
from plotoptix.enums import *
from copy import deepcopy
from plotoptix.materials import m_clear_glass, m_mirror, m_plastic, m_transparent_plastic
m_mirror["VarFloat3"]["survace_albedo"] = [0.9,0.9,0.9]
ground_plastic = deepcopy(m_plastic)
ground_plastic["VarFloat3"]["surface_albedo"] = [0,0,0]
m_clear_glass["VarFloat3"]["refraction_index"] = [1.1, 1.1, 1.1]
m_clear_glass["VarFloat3"]["surface_albedo"] = [2.0, 2.0, 2.0]
from tet_mesh import tet2obj
import trimesh
from pathlib import Path
cone_mesh = trimesh.load(Path("asset") / "thin_cone.obj")
# m_clear_glass["VarFloat3"]["refraction_index"] = [1.4, 1.44, 1.5]

ROOT = 'E:\\ACG_repo_2\\npys\\bunny_in_pipe\\'

filename_list = list()
for root, dirs, files in os.walk(ROOT):
    for file in files:
         # 0001.npy.
        if file.endswith('.npy') and file.startswith('0'):
            filename_list.append(os.path.join(root, file))
            
obj1_list = list()
for filename in filename_list:
    try:
        obj1_list.append((np.load(filename),np.load(os.path.join(ROOT,"elements.npy"))))
    except Exception as e:
        print(e)
        pass
N = len(obj1_list[0][0])
print("N=",N)
vertices, face_idx = tet2obj(obj1_list[0][0], obj1_list[0][1])
obj2_list = [obj1_list_item[0] for obj1_list_item in obj1_list]
obj1_list = obj2_list
class params():
    fps = 30
    duration = 7
    dt = 2*np.pi / (fps*duration)
    t = 0.0
    n = 0
    obj1_pos = obj1_list[0]

def compute(rt: NpOptiX, delta: int) -> None:
    params.t += params.dt
    params.n += 1
    
def update(rt: NpOptiX)->None:
    print(params.n)
    rt.update_mesh("obj1", pos = obj1_list[params.n])

def redraw(rt):
    imgplot.set_data(rt._img_rgba)
    plt.draw()
    
width = 960; height = 540
plt.figure(1,figsize=(9.5,5.5))
plt.tight_layout()
imgplot = plt.imshow(np.zeros((height, width, 4), dtype=np.uint8))

optix = NpOptiX(on_scene_compute=compute,
                on_rt_completed=update,
                on_launch_finished=redraw,
                width=width, height=height,
                start_now=False
)
optix.setup_material("glass", m_clear_glass)
optix.setup_material("plastic", m_plastic)
optix.setup_material("mirror", m_mirror)
optix.setup_material("ground_plastic", ground_plastic)

#optix.encoder_create(fps=15, bitrate=8)
optix.encoder_create(fps=30, bitrate=4, profile="High")
#optix.encoder_create(fps=15, profile="High444", preset="Lossless")

optix.set_param(min_accumulation_step=128,    # 1 animation frame = 128 accumulation frames
                max_accumulation_frames=512)  # accumulate 512 frames when paused
optix.set_uint("path_seg_range", 5, 10)

circ_array_list = []
for N_ in range(1000):
    theta = 2*np.pi * N_ / 1000
    circ_array_list.append([0.5*np.cos(theta), 0.5*np.sin(theta), 1.5])

circ_array_list_2 = []
for N_ in range(1000):
    theta = 2*np.pi * N_ / 1000
    circ_array_list_2.append([0.55*np.cos(theta), 0.55*np.sin(theta), 1.5])

cone_array_list = [[0,0,0]]+circ_array_list+circ_array_list_2
cone_faces_list = []
for N_ in range(1000):
    cone_faces_list.append([0,1+(N_%1000),1+(N_+1)%1000])
for N_ in range(1000):
    cone_faces_list.append([0,1001+(N_+1)%1000,1001+(N_%1000)])
for N_ in range(1000):
    cone_faces_list.append([1+(N_%1000),1001+(N_%1000),1001+(N_+1)%1000])
    cone_faces_list.append([1+(N_%1000),1+(N_+1)%1000,1001+(N_+1)%1000])
    
cone_array = np.array(cone_array_list)*1.5
print(cone_array.shape)
    





exposure = 0.8; gamma = 2.2
optix.set_float("tonemap_exposure", exposure) # sRGB tonning
optix.set_float("tonemap_gamma", gamma)
# optix.set_float("denoiser_blend", 0.0)
# optix.add_postproc("Gamma") 
optix.add_postproc("Denoiser")

optix.set_ambient([0.01, 0.02, 0.03])
optix.set_background(0)

optix.set_mesh("obj1", pos=obj1_list[0],  c=make_color(0.8, 0.8, 0.8),mat='plastic',make_normals=True,faces=face_idx)
# optix.set_mesh("cone_1", pos = cone_mesh.vertices*1e3, c=make_color(0.2, 0.7, 0.3),mat='mirror',make_normals=True,faces=cone_mesh.faces)
# optix.set_mesh("cone_2", pos = cone_mesh.vertices*1e3 + np.array([0.5, 0, 0]), c=make_color(0.2, 0.7, 0.3),mat='mirror',make_normals=True,faces=cone_mesh.faces)
# optix.set_mesh("cone_3", pos = cone_mesh.vertices*1e3 + np.array([-0.5, 0, 0]), c=make_color(0.2, 0.7, 0.3),mat='mirror',make_normals=True,faces=cone_mesh.faces)
# optix.set_mesh("cone_4", pos = cone_mesh.vertices*1e3 + np.array([0, 0.5, 0]), c=make_color(0.2, 0.7, 0.3),mat='mirror',make_normals=True,faces=cone_mesh.faces)
# optix.set_mesh("cone_5", pos = cone_mesh.vertices*1e3 + np.array([0, -0.5, 0]), c=make_color(0.2, 0.7, 0.3),mat='mirror',make_normals=True,faces=cone_mesh.faces)
# optix.set_mesh("cone_6", pos = cone_mesh.vertices*1e3 + np.array([0.5, -0.5, 0]), c=make_color(0.2, 0.7, 0.3),mat='mirror',make_normals=True,faces=cone_mesh.faces)
# optix.set_mesh("cone_7", pos = cone_mesh.vertices*1e3 + np.array([-0.5, -0.5, 0]), c=make_color(0.2, 0.7, 0.3),mat='mirror',make_normals=True,faces=cone_mesh.faces)
# optix.set_mesh("cone_8", pos = cone_mesh.vertices*1e3 + np.array([-0.5, 0.5, 0]), c=make_color(0.2, 0.7, 0.3),mat='mirror',make_normals=True,faces=cone_mesh.faces)
# optix.set_mesh("cone_9", pos = cone_mesh.vertices*1e3 + np.array([0.5, 0.5, 0]), c=make_color(0.2, 0.7, 0.3),mat='mirror',make_normals=True,faces=cone_mesh.faces)
optix.set_mesh("mirror_1",pos = np.array([[3.8,-0.9,5],[-0.9,-0.9,5],[-0.9,-0.9,1],[3.8,-0.9,1]]),c = make_color(0.2,0.7,0.3),mat="mirror",make_normals=True,faces=np.array([[0,1,2],[2,3,0]]))
optix.set_mesh("mirror_2",pos = np.array([[-0.9,3.8,5],[-0.9,-0.9,5],[-0.9,-0.9,1],[-0.9,3.8,1]]),c = make_color(0.2,0.7,0.3),mat="mirror",make_normals=True,faces=np.array([[0,1,2],[2,3,0]]))
optix.set_mesh("cone", pos = cone_array, c=10,mat='glass',make_normals=True,faces=cone_faces_list)
# optix.set_mesh("ground",pos = np.array([[0,1000,-2],[-1000,-1000,-2],[1000,-1000,-2]]),c = make_color(np.array([0.04,0.01,0.03]), 0.7, 0.7),mat="plastic",make_normals=True,faces=np.array([[0,1,2]]))
# optix.set_mesh("ground2",pos = np.array([[-10,-10,-1000],[-10+1000,-10-1000,1000],[-10-1000,-10+1000,1000]]),c = make_color(np.array([0.1,0.1,0.3]), 0.7, 0.7),mat="ground_plastic",make_normals=True,faces=np.array([[0,1,2]]))
# optix.set_mesh("ground3",pos = np.array([[0,50,-1000],[-1000,50,1000],[1000,50,1000]]),c = make_color(np.array([0.1,0.1,0.3]), 0.7, 0.7),mat="ground_plastic",make_normals=True,faces=np.array([[0,1,2]]))
optix.setup_camera("cam1", cam_type="Pinhole",
                   eye=[3.5,3.5, 3.2], target=[0, 0, 2.1], up=[0, 0, 1],
                   aperture_radius=0.0,
                   fov=35)

optix.setup_light("light1", pos=[15, 12, 15], color=[120/1.4,30/1.4,90/1.4], radius=2)
optix.setup_light("light2", pos=[-15, 12, 15], color=[120/1.4,30/1.4,90/1.4], radius=2)
optix.setup_light("light3", pos=[15, -12, 15], color=[120/1.4,30/1.4,90/1.4], radius=2)
optix.setup_light("light4", pos=[0, 0, 15], color=[120/1.4,30/1.4,90/1.4], radius=2)
# optix.setup_light("light5", pos=[0,0, -30], color=[120/1.1,30/1.1,90/1.1], radius=2)
# optix.setup_light("light1234", pos=[15, 12, -6], color=[120/1.4,30/1.4,90/1.4], radius=2)
optix.setup_light("light42543", pos=[-15, 12, -6], color=[120/1.4,30/1.4,90/1.4], radius=2)
# optix.setup_light("light123", pos=[15, -12, -6], color=[120/1.4,30/1.4,90/1.4], radius=2)
optix.setup_light("lighterw4", pos=[0, 0, 0.2], color=[120/1.4,30/1.4,90/1.4], radius=0.2)
optix.setup_light("par",light_type="Parallelogram",color=[120/1.4,30/1.4,90/1.4])
optix.start()

optix.encoder_start("bunnyInPipe_02.mp4", params.fps * params.duration)
import time
n = 0
while n < 400:
    n+= 1
    time.sleep(1)
    print(optix.encoding_frames(), optix.encoded_frames())
    if optix.encoded_frames() == optix.encoding_frames():
        n += 114514

optix.encoder_stop()

optix.close()