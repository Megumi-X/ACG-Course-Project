import os
import matplotlib.pyplot as plt
import numpy as np
from plotoptix import NpOptiX
from plotoptix.utils import make_color, map_to_colors
from plotoptix.enums import *
from plotoptix.materials import m_clear_glass, m_mirror, m_plastic
from tet_mesh import tet2obj
import trimesh
from pathlib import Path
cone_mesh = trimesh.load(Path("asset") / "thin_cone.obj")
m_clear_glass["VarFloat3"]["refraction_index"] = [1.4, 1.44, 1.5]

ROOT = 'E:\\ACG_repo_2\\npys\\bunny\\'

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
    duration = 6
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

#optix.encoder_create(fps=15, bitrate=8)
optix.encoder_create(fps=30, bitrate=4, profile="High")
#optix.encoder_create(fps=15, profile="High444", preset="Lossless")

optix.set_param(min_accumulation_step=128,    # 1 animation frame = 128 accumulation frames
                max_accumulation_frames=512)  # accumulate 512 frames when paused
optix.set_uint("path_seg_range", 5, 10)

exposure = 0.8; gamma = 2.2
optix.set_float("tonemap_exposure", exposure) # sRGB tonning
optix.set_float("tonemap_gamma", gamma)
optix.set_float("denoiser_blend", 0.0)
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
optix.set_mesh("ground",pos = np.array([[0,1000,0],[-1000,-1000,0],[1000,-1000,0]]),c = make_color(0,255,0),mat="plastic",make_normals=True,faces=np.array([[0,1,2]]))


optix.setup_camera("cam1", cam_type="Pinhole",
                   eye=[4.5,4.5, 0.8], target=[0, 0, 0.8], up=[0, 0, 1],
                   aperture_radius=0.0,
                   fov=35)

optix.setup_light("light1", pos=[15, 12, 15], color=[120,30,90], radius=2)
optix.setup_light("light2", pos=[-15, 12, 15], color=[120,30,90], radius=2)
optix.setup_light("light3", pos=[15, -12, 15], color=[120,30,90], radius=2)
optix.setup_light("light4", pos=[-15, -12, 15], color=[120,30,90], radius=2)

optix.start()

optix.encoder_start("bunny_02.mp4", params.fps * params.duration)
import time
n = 0
while n < 200:
    n+= 1
    time.sleep(1)
    print(optix.encoding_frames(), optix.encoded_frames())
    if optix.encoded_frames() == optix.encoding_frames():
        n += 114514

optix.encoder_stop()

optix.close()