from pathlib import Path
import os

import numpy as np

from multiprocessing import Pool

root = "./"
from pbrt_renderer import create_folder, to_real_array
from pbrt_renderer import PbrtRenderer
from tet_mesh import tet2obj
import imageio

def export_gif(folder_name, gif_name, fps, name_prefix, name_suffix):
    frame_names = [os.path.join(folder_name, f) for f in os.listdir(folder_name)
        if os.path.isfile(os.path.join(folder_name, f)) and f.startswith(name_prefix) and f.endswith(name_suffix)]
    frame_names = sorted(frame_names)

    # Read images.
    images = [imageio.v2.imread(f) for f in frame_names]
    if fps > 0:
        imageio.mimsave(gif_name, images, fps=fps)
    else:
        imageio.mimsave(gif_name, images)

def render_data(image_name, obj):
    spp = 64

    r = PbrtRenderer()
    eye = to_real_array([4, 4, 2.0])
    look_at = to_real_array([0.0, 0.0, 1.3])
    eye = look_at + 0.8 * (eye - look_at)
    r.set_camera(eye=eye, look_at=look_at, up=[0, 0, 1], fov=45)
    r.add_infinite_light({
        "rgb L": (0.6, 0.7, 0.8)
    })
    r.add_spherical_area_light([30, 50, 60], 3, [1, 1, 1], 1e5)

    # Convert voxels into surface triangle meshes.
    # If you use tet meshes, this step can be simplified by calling tet2obj from tet_mesh.
    # elements = []
    # for e in range(obj[1]):
    #     elements.append(e[0], e[1], e[2])
    #     elements.append(e[0], e[1], e[3])
    #     elements.append(e[0], e[2], e[3])
    #     elements.append(e[1], e[2], e[3])
    
    # r.add_triangle_mesh(obj[0], elements, None, None, ("diffuse", { "rgb reflectance": (0.1, 0.4, 0.7) }))

    vertices, elements = tet2obj(obj[0], obj[1])
    
    r.add_triangle_mesh(vertices, elements, None, None, ("diffuse", { "rgb reflectance": (0.1, 0.4, 0.7) }))
    r.add_plane([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 10000, ("diffuse", { "rgb reflectance": (0.7, 0.4, 0.1) }))

    # The real rendering job starts here.
    r.set_image(pixel_samples=spp, file_name=image_name,
        resolution=[600, 600])
    r.render(use_gpu=True)
def render_data_wrapper(arg):
    return render_data(arg[0],arg[1])
def main():
    data_folder = Path(root) / "bunny_pressure"
    render_folder = Path(root) / "render_bunny_pressure"
    create_folder(render_folder, exist_ok=True)

    for f in range(0, 250):
        obj = (np.load(data_folder / "{:04d}.npy".format(f)), np.load(data_folder / "elements.npy"))
        render_data(render_folder / "{:04d}.png".format(f), obj)
    export_gif(render_folder, render_folder / "bunny.gif", 0, "0", ".png")
if __name__ == "__main__":
    main()