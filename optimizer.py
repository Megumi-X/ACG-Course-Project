import taichi as ti
from copy import deepcopy
ti.init()
@ti.func
def test_E(x: ti.template()):
    sum_ = 0.0
    for i in range(4):
        for j in range(4):
            sum_ += x[i,j]**2
    return sum_


@ti.func
def test_jac(x: ti.template()):
    return 2*x
    
@ti.func
def norm(x:ti.template()):
    sum_ = 0.0
    for i in range(4):
        for j in range(4):
            sum_ += x[i,j]**2
    return sum_**0.5

@ti.func
def minimizer(init_pos:ti.template()):
    options = dict(
        ftol = 1e-20,
        maxiter = 600,
    )
    ftol = options["ftol"]
    maxiter = options["maxiter"]
    # assert method in ["Adam"]
    b1 = 0.9
    b2 = 0.99
    lr = 1e-2
    m = test_jac(init_pos)
    v = m**2
    old_pos = deepcopy(init_pos)
    new_pos = deepcopy(init_pos)
    counter = 0
    ti.loop_config(serialize=True)
    for _ in range(maxiter):
        old_E = test_E(old_pos)
        
        m += -m + b1 * m + (1-b1) * test_jac(new_pos)
        v += -v + b2 * v + (1-b2) * test_jac(new_pos)**2
        new_pos -= lr*m / (v**0.5 + 1e-13)
        
        
        new_E = test_E(new_pos)
        old_pos -= lr*m / (v**0.5 + 1e-13)
        if abs(new_E-old_E)/abs(old_E) < ftol:
            break
        if abs(new_E-old_E) < ftol:
            break
        
    return new_pos




@ti.kernel
def min_x(x:ti.template()):
    y = minimizer(x)
    print(y)

@ti.kernel
def init_x(x:ti.template()):
    for i in range(x.n):
        for j in range(x.m):
            x[i][j] = 0.1
if __name__ == '__main__':
    X = ti.types.matrix(4,4,ti.f32)
    x = X([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,-0.1,-0.2,-0.3,-0.4,0.5,0.6,0.7])
    # init_x(x)
    
    print(x)
    min_x(x)
    