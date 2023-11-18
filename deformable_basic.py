import taichi as ti
from domain import Domain

if __name__ == '__main__':
    ti.init(arch=ti.cuda)

@ti.dataclass
class Material:
    density: ti.f32
    youngs_modulus: ti.f32
    poissons_ratio: ti.f32

@ti.data_oriented
class GeometryShape:
    @ti.kernel
    def __init__(self):
        self.undeformed = Domain()
        
