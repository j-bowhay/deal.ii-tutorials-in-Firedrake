import firedrake as fd
import matplotlib.pyplot as plt

# create mesh
mesh = fd.RectangleMesh(nx=32, ny=32, Lx=1, Ly=1, originX=-1, originY=-1,
                        quadrilateral=True)

# define function spaces
V = fd.FunctionSpace(mesh, "CG", 1)
u = fd.TrialFunction(V)
v = fd.TestFunction(V)

# variational problem, BCs are weakly enforced
a = fd.inner(fd.grad(u), fd.grad(v)) * fd.dx
L = fd.Constant(1) * v * fd.dx
bc = fd.DirichletBC(V, fd.Constant(0), "on_boundary")

u = fd.Function(V, name="solution")
fd.solve(a == L, u, bcs=bc, solver_parameters={'ksp_type': 'cg',
                                               'ksp_monitor_true_residual': None})

im = fd.tripcolor(u)
plt.colorbar(im)
plt.show()
