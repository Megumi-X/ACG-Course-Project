import taichi as ti
ti.init()
x = ti.Matrix.field([[0.0 for i in range(4)] for j in range(3)])
@ti.kernel
def initialize():
    for i in range(3):
        for j in range(4):
            x[i,j] = 1.0

if __name__ == '__main__':
    initialize()
    print(x)