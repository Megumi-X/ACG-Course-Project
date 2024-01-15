import os
import matplotlib.pyplot as plt
import numpy as np
from plotoptix import NpOptiX
from plotoptix.utils import make_color, map_to_colors
from plotoptix.enums import *
from plotoptix.materials import m_clear_glass, m_mirror, m_plastic
m_clear_glass["VarFloat3"]["refraction_index"] = [1.4, 1.44, 1.5]

ROOT = 'E:\\ACG_repo_2\\ACG-Course-Project-Pytorch\\simple_cloth----\\'
X = 30
Y = 50

filename_list = list()
for root, dirs, files in os.walk(ROOT):
    for file in files:
         # 0001.npy.
        if file.endswith('.npy') and file.startswith('0'):
            filename_list.append(os.path.join(root, file))


# def preprocess(numpy_array):
#     # Sorting the numpy array first by the 0th column, then 1st and then 2nd
#     # Argsort generates indices that would sort the array, and it's applied across the rows
#     sorted_indices = np.lexsort((numpy_array[:, 0], numpy_array[:, 2], numpy_array[:, 1]))
#     sorted_array = numpy_array[sorted_indices]

#     return sorted_array, sorted_indices
# def recreate_sorted_array(next_array, sorted_indices):
#     # Invert the sorted indices to map from original to sorted positions
#     return next_array[sorted_indices]
# new_0, sorted_indices = preprocess(np.load(filename_list[0])*6)
obj1_list = list()
for filename in filename_list:
    try:
        obj1_list.append(np.load(filename)*6)
    except Exception as e:
        print(e)
        pass
N = len(obj1_list[0])
print("N=",N)


class params():
    fps = 30
    duration = 10
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

#optix.encoder_create(fps=20, bitrate=8)
optix.encoder_create(fps=30, bitrate=4, profile="High")
#optix.encoder_create(fps=20, profile="High444", preset="Lossless")

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
face_list_list = list()
# for idx in range(3000):
#     face_list_list.append([idx,idx+1,idx+2])
    
    
for i in range(0,Y+1,1):
    for j in range(0,X+1,1):
        if i > 0 and i < Y and j > 0 and j < X:
            face_list_list.append(
                [(i*(X+1)+j)*2, ((i+1)*(X+1)+j)*2, (i*(X+1)+j+1)*2]
            )
            face_list_list.append(
                [(i*(X+1)+j+1)*2, ((i+1)*(X+1)+j)*2, ((i+1)*(X+1)+j+1)*2]
            )
            # face_list_list.append(
            #     [(i*(X+1)+j)*2+1, ((i+1)*(X+1)+j)*2+1,(i*(X+1)+j+1)*2+1]
            # )
            # face_list_list.append(
            #     [(i*(X+1)+j+1)*2+1, ((i+1)*(X+1)+j)*2+1,((i+1)*(X+1)+j+1)*2+1]
            # )

    
    
faces_array = np.array(face_list_list)
print(faces_array.shape)
print(faces_array.max())
optix.set_mesh("obj1", pos=obj1_list[0],  c=make_color(0.8, 0.8, 0.8),mat='plastic',make_normals=True,faces=faces_array)

optix.setup_camera("cam1", cam_type="Pinhole",
                   eye=[13, 18, 7], target=[0, 7, 0], up=[0, 0, 1],
                   aperture_radius=0.0,
                   fov=60)

optix.setup_light("light1", pos=[15, 12, 15], color=[120,30,90], radius=2)
optix.setup_light("light2", pos=[15, 12, -15], color=[30,120,90], radius=2)
optix.start()

optix.encoder_start("simple_cloth_3.mp4", params.fps * params.duration)

import time
n = 0
while n < 1000:
    n+= 1
    time.sleep(1)
    print(optix.encoding_frames(), optix.encoded_frames())
    if optix.encoded_frames() == optix.encoding_frames():
        n+= 114514

optix.encoder_stop()

optix.close()