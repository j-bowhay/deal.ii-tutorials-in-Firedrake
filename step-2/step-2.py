import firedrake as fd
import scipy.sparse as sp
import matplotlib.pyplot as plt
from petsc4py import PETSc

# first we will generate the annulus mesh again
annulus = fd.AnnulusMesh(1, 0.5, nt=20)

# define function space and bilinear form
V = fd.FunctionSpace(annulus, "CG", 1)
u = fd.TrialFunction(V)
v = fd.TestFunction(V)
a = fd.inner(fd.grad(u), fd.grad(v)) * fd.dx

# assemble stiffness matrix
petsc_mat_original = fd.assemble(a).petscmat

# get the original stiffness matrix
indptr, indices, data = petsc_mat_original.getValuesCSR()
original = sp.csr_matrix((data, indices, indptr), shape=petsc_mat_original.getSize())

# permute the stiffness matrix via Reverse Cuthill-McKee
perm = petsc_mat_original.getOrdering(PETSc.Mat.OrderingType.RCM)[0]
petsc_mat_perm = petsc_mat_original.permute(perm, perm)
indptr, indices, data = petsc_mat_perm.getValuesCSR()
permuted = sp.csr_matrix((data, indices, indptr), shape=petsc_mat_perm.getSize())

# plot the sparsity patterns
fig, axs = plt.subplots(1, 2)
axs[0].spy(original)
axs[1].spy(permuted)
plt.show()