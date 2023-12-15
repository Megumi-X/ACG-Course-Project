##########################################################################################################################
# This file is a citation from the following repo: https://github.com/taodu-eecs/GraDy/blob/master/python/tet_mesh.py
# The original authors are Prof.Du and other contributors of the GraDy project.
# The citation has got permission from the original authors.
##########################################################################################################################

from pathlib import Path, PosixPath
import os

import numpy as np
import pathlib
import shutil

from tet_mesh import to_real_array, to_integer_array
root = "./"

np_real = np.float64
np_integer = np.int32

def create_folder(folder_name, exist_ok):
    pathlib.Path(folder_name).mkdir(parents=True, exist_ok=exist_ok)

def delete_folder(folder_name):
    shutil.rmtree(folder_name)

def delete_file(file_name):
    pathlib.Path(file_name).unlink()

class PbrtRenderer:
    def __init__(self):
        # Camera parameters.
        # For now, we assume a perspective camera is used.
        self.__eye = None
        self.__look_at = None
        self.__up = None
        self.__fov = None

        # Image related parameters.
        self.__pixel_samples = None
        self.__file_name = None
        self.__resolution = None

        # Lights.
        self.__lights = []

        # Area lights.
        self.__area_lights = []

        # Shapes.
        self.__shapes = []

    def set_camera(self, eye, look_at, up, fov):
        self.__eye = to_real_array(eye).ravel()
        self.__look_at = to_real_array(look_at).ravel()
        self.__up = to_real_array(up).ravel()
        self.__fov = np_real(fov)

    # resolution = [width, height].
    def set_image(self, pixel_samples, file_name, resolution):
        self.__pixel_samples = np_integer(pixel_samples)
        self.__file_name = str(file_name)
        self.__resolution = to_integer_array(resolution).ravel()

    def add_infinite_light(self, light_properties):
        self.__lights.append(("infinite", light_properties))

    # rgb is between 0 and 1.
    def add_distant_light(self, from_point, to_point, rgb):
        self.__lights.append(("distant", {
            "point3 from": to_real_array(from_point).ravel(),
            "point3 to": to_real_array(to_point).ravel(),
            "rgb L": to_real_array(rgb).ravel()
        }))

    # For now, we only support spherical area light.
    def add_spherical_area_light(self, center, radius, rgb, power, transforms=[]):
        self.__area_lights.append((center, radius, rgb, power, transforms))

    def clear_lights(self):
        self.__lights = []

    # material = (material_type: str, material_properties: dict).
    # transforms: list of (str, real np.array).
    def add_sphere(self, center, radius, material, transforms=[], alpha=1):
        self.__shapes.append((
            "sphere",
            {
                "float radius": np_real(radius),
                "float alpha": np_real(alpha)
            },
            None,
            material,
            transforms + [("Translate", to_real_array(center).ravel()),]
        ))

    def add_cylinder(self, bottom_center, top_center, radius, material, transforms=[], alpha=1):
        axis = top_center - bottom_center
        axis /= np.linalg.norm(axis)
        # Rotate z to axis.
        rot_axis = np.cross([0, 0, 1], axis)
        rot_axis_norm = np.linalg.norm(rot_axis)
        rot_angle = np.rad2deg(np.arccos(axis[2]))
        # Fix corner case.
        if rot_axis_norm < 1e-6:
            # Corner cases.
            if axis[2] > 0:
                rot_axis = to_real_array([1, 0, 0])
                rot_angle = 0
            else:
                rot_axis = np.cross(axis, np.random.normal(size=3))
                rot_axis /= np.linalg.norm(rot_axis)
                rot_angle = 180

        self.__shapes.append((
            "cylinder",
            {
                "float radius": np_real(radius),
                "float zmin": np_real(0),
                "float zmax": np.linalg.norm(top_center - bottom_center),
                "float alpha": np_real(alpha)
            },
            None,
            material,
            transforms + [
                ("Translate", bottom_center),
                ("Rotate", (rot_angle, rot_axis[0], rot_axis[1], rot_axis[2]))
            ]
        ))

    def add_plane(self, center, normal, size, material, texture_image=None, transforms=[], alpha=1):
        center = to_real_array(center).ravel()
        normal = to_real_array(normal).ravel()
        size = np_real(size)
        x = to_real_array(np.random.rand(3))
        x = np.cross(normal, x)
        y = np.cross(normal, x)
        # Now x, y, normal form a right-hand frame.
        x = x / np.linalg.norm(x)
        y = y / np.linalg.norm(y)
        v00 = center + x * -size / 2 + y * -size / 2
        v01 = center + x * -size / 2 + y * size / 2
        v10 = center + x * size / 2 + y * -size / 2
        v11 = center + x * size / 2 + y * size / 2
        vertices = to_real_array([v00, v01, v10, v11])
        elements = to_integer_array([[0, 2, 1], [1, 2, 3]])
        textures = to_real_array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.add_triangle_mesh(vertices, elements, textures, texture_image, material, transforms, alpha)

    def add_curve(self, points, deg, width, material, transforms=[], alpha=1):
        self.__shapes.append((
            "curve",
            {
                "point3 P": to_real_array(points).ravel(),
                "integer degree": np_integer(deg),
                "float width": np_real(width),
                "float alpha": np_real(alpha)
            },
            None,
            material,
            transforms
        ))

    # If there is no texture, simply set texture_coords=None or texture_image=None.
    def add_triangle_mesh(self, vertices, elements, texture_coords, texture_image,
        material, transforms=[], alpha=1):
        if texture_coords is None or texture_image is None:
            self.__shapes.append((
                "trianglemesh",
                {
                    "integer indices": to_integer_array(elements).ravel(),
                    "point3 P": to_real_array(vertices).ravel(),
                    "float alpha": np_real(alpha)
                },
                None,
                material,
                transforms
            ))
        else:
            self.__shapes.append((
                "trianglemesh",
                {
                    "integer indices": to_integer_array(elements).ravel(),
                    "point3 P": to_real_array(vertices).ravel(),
                    "point2 uv": to_real_array(texture_coords).ravel(),
                    "float alpha": np_real(alpha)
                },
                str(texture_image),
                material,
                transforms
            ))

    def clear_shapes(self):
        self.__shapes = []

    def render(self, use_gpu):
        # Generate a temporary folder.
        tmp_folder = Path(root) / ".pbrt"
        create_folder(tmp_folder, exist_ok=True)

        # Create a .pbrt file.
        with open(tmp_folder / "scene.pbrt", "w") as f:
            # Camera info.
            # Note the minus signs --- they are used to swap handness.
            f.write("LookAt {} {} {}\n".format(self.__eye[0], self.__eye[1], -self.__eye[2]))
            f.write("       {} {} {}\n".format(self.__look_at[0], self.__look_at[1], -self.__look_at[2]))
            f.write("       {} {} {}\n".format(self.__up[0], self.__up[1], -self.__up[2]))
            f.write("Camera \"perspective\" \"float fov\" {}\n".format(self.__fov))
            f.write("\n")

            # Image info.
            f.write("Sampler \"halton\" \"integer pixelsamples\" {}\n".format(self.__pixel_samples))
            f.write("Integrator \"volpath\"\n")
            f.write("Film \"rgb\" \"string filename\" \"{}\"\n".format(self.__file_name))
            f.write("   \"integer xresolution\" [{}] \"integer yresolution\" [{}]\n".format(
                self.__resolution[0], self.__resolution[1]))
            f.write("\n")

            f.write("WorldBegin\n")
            f.write("\n")

            # Swap handness: pbrt uses left-hand coordinates while we use right-hand coordinates.
            f.write("AttributeBegin\n")
            f.write("Scale 1 1 -1\n")
            f.write("\n")

            def convert_value_to_str(value):
                if type(value) in [str, PosixPath]:
                    return "\"{}\"".format(str(value))
                elif type(value) in [float, int, np_real, np_integer]:
                    return str(value)
                else:
                    # It has to be a 1D numpy array.
                    value = list(value)
                    is_float = False
                    for v in value:
                        if type(v) in [float, np_real]:
                            is_float = True
                            break
                        else:
                            assert type(v) in [int, np_integer]
                    if is_float:
                        value = to_real_array(value).ravel()
                    else:
                        value = to_integer_array(value).ravel()
                    return "[" + " ".join([str(v) for v in value]) + "]"

            # Lighting.
            for light_type, light_properties in self.__lights:
                f.write("LightSource \"{}\"\n".format(light_type))
                for k, v in light_properties.items():
                    f.write("   \"{}\" {}\n".format(k, convert_value_to_str(v)))
                f.write("\n")
            for center, radius, rgb, power, transforms in self.__area_lights:
                f.write("AttributeBegin\n")
                f.write("   AreaLightSource \"diffuse\" \"rgb L\" [{} {} {}] \"float power\" [ {} ]\n".format(
                    rgb[0], rgb[1], rgb[2], power
                ))
                for k, v in transforms:
                    f.write("   {} {}\n".format(k, " ".join([str(vv) for vv in v])))
                f.write("   Translate {} {} {}\n".format(center[0], center[1], center[2]))
                f.write("   Shape \"sphere\" \"float radius\" {}\n".format(radius))
                f.write("AttributeEnd\n")
                f.write("\n")

            # Shapes.
            # We use {:08d} to create names used in pbrt. Therefore, we have the following assert.
            assert len(self.__shapes) < 1e8
            for shape_idx, (shape_type, shape_properties, texture_image, material, transforms) in enumerate(self.__shapes):
                f.write("AttributeBegin\n")
                if texture_image is not None:
                    # Material with textures.
                    f.write("   Texture \"texture_{:08d}\"\n".format(shape_idx))
                    f.write("       \"spectrum\" \"imagemap\" \"string filename\" \"{}\"\n".format(texture_image))

                    material_type, material_properties = material
                    f.write("   Material \"{}\"\n".format(material_type))
                    for k, v in material_properties.items():
                        if "reflectance" in k: continue
                        f.write("       \"{}\" {}\n".format(k, convert_value_to_str(v)))
                    f.write("       \"texture reflectance\" \"texture_{:08d}\"\n".format(shape_idx))
                else:
                    # Material without textures.
                    material_type, material_properties = material
                    f.write("   Material \"{}\"\n".format(material_type))
                    for k, v in material_properties.items():
                        f.write("       \"{}\" {}\n".format(k, convert_value_to_str(v)))
                # Transforms.
                for k, v in transforms:
                    f.write("   {} {}\n".format(k, " ".join([str(vv) for vv in v])))
                # Shape.
                f.write("   Shape \"{}\"\n".format(shape_type))
                for k, v in shape_properties.items():
                    f.write("       \"{}\" {}\n".format(k, convert_value_to_str(v)))
                f.write("AttributeEnd\n")
                f.write("\n")

            f.write("AttributeEnd\n")

        # Call pbrt.
        if use_gpu:
            os.system("{} --gpu {}".format("pbrt", tmp_folder / "scene.pbrt"))
        else:
            os.system("{} {}".format(Path(root) / "build" / "pbrt-v4" / "pbrt", tmp_folder / "scene.pbrt"))

        # Delete the tempory folder.
        delete_folder(tmp_folder)

if __name__ == "__main__":
    file_name = Path(root) / "python" / "image.png"

    r = PbrtRenderer()

    # Camera information.
    r.set_camera(eye=[3, 4, 1.5], look_at=[.5, .5, 0], up=[0, 0, 1], fov=45)

    # Image information.
    r.set_image(pixel_samples=128, resolution=[400, 400], file_name=file_name)

    # Lighting.
    r.add_infinite_light({ "rgb L": [.4, .45, .5] })
    r.add_distant_light(from_point=[-30, 40, 100], to_point=[0, 0, 1], rgb=[.3, .2, .1])

    # Shapes.
    r.add_sphere(center=[0, 0, 0], radius=1, material=("dielectric", {}))
    r.add_plane(center=[0, 0, -1], normal=[0, 0, 1], size=1000, material=(
        "diffuse", {
            "rgb reflectance": [.2, .33, .2]
        }
    ))

    r.render(use_gpu=False)

    # Show the result.
    os.system("eog {}".format(file_name))
    print("Press enter to continue...")
    input()

    delete_file(file_name)