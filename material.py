import taichi as ti

# Here we use neo-Hookean model

@ti.kernel
def DeterminantGrad(A: ti.types.matrix(3, 3, ti.f32)) -> ti.types.matrix(3, 3, ti.f32):
    dJdA = ti.Matrix([[0.0 for i in range(3)] for j in range(3)])
    dJdA[0, 0] = A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]
    dJdA[0, 1] = A[0, 2] * A[2, 1] - A[0, 1] * A[2, 2]
    dJdA[0, 2] = A[0, 1] * A[1, 2] - A[0, 2] * A[1, 1]
    dJdA[1, 0] = A[1, 2] * A[2, 0] - A[1, 0] * A[2, 2]
    dJdA[1, 1] = A[0, 0] * A[2, 2] - A[0, 2] * A[2, 0]
    dJdA[1, 2] = A[0, 2] * A[1, 0] - A[0, 0] * A[1, 2]
    dJdA[2, 0] = A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0]
    dJdA[2, 1] = A[0, 1] * A[2, 0] - A[0, 0] * A[2, 1]
    dJdA[2, 2] = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    return dJdA    

@ti.data_oriented
class Material:
    def __init__(self):
        self.density = 0.0
        self.youngs_modulus = 0.0
        self.poissons_ratio = 0.0
        self.lam = 0.0
        self.mu = 0.0

    @ti.kernel
    def Initialize(self, density: ti.f32, youngs_modulus: ti.f32, poissons_ratio: ti.f32):
        self.density = density
        self.youngs_modulus = youngs_modulus
        self.poissons_ratio = poissons_ratio
        self.lam = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
        self.mu = youngs_modulus / (2 * (1 + poissons_ratio))

    @ti.kernel
    def ComputeEnergyDensity(self, F: ti.types.matrix(3, 3, ti.f32)) -> ti.f32:
        C = F.transpose() @ F
        J = F.determinant()
        Ic = C[0, 0] + C[1, 1] + C[2, 2]
        delta = 1
        la = self.lam
        mu = self.mu
        dim = 3
        alpha = (1 - 1 / (dim + delta)) * mu / la + 1
        return mu / 2 * (Ic - dim) + la / 2 * (J - alpha) * (J - alpha) - 0.5 * mu * ti.log(Ic + delta)

    @ti.kernel
    def ComputeStressDensity(self, F: ti.types.matrix(3, 3, ti.f32)) -> ti.types.matrix(3, 3, ti.f32):
        C = F.transpose() @ F
        J = F.determinant()
        Ic = C[0, 0] + C[1, 1] + C[2, 2]
        delta = 1
        la = self.lam
        mu = self.mu
        dim = 3
        alpha = (1 - 1 / (dim + delta)) * mu / la + 1
        dJdF = DeterminantGrad(F)
        return (1 - 1 / (Ic + delta)) * mu * F + la * (J - alpha) * dJdF