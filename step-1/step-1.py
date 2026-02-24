import firedrake as fd
import matplotlib.pyplot as plt

# the first mesh discussed in the tutorial is a unit square
square_mesh = fd.UnitSquareMesh(4**2, 4**2, quadrilateral=True)
fd.triplot(square_mesh)
plt.show()

# the second mesh is an annulus, Firedrakes's built in mesh generation doesn't allow
# for the adaptivity used in the deal.ii example

annulus = fd.AnnulusMesh(1, 0.5, nt=20)
# Firedrake's built-in plotting doesn't support extruded meshes so we save to a file
# and view in ParaView
fd.VTKFile("step-1/annulus_mesh.pvd").write(annulus)

# triangle mesh
triangle_mesh = fd.UnitTriangleMesh(refinement_level=3)
fd.triplot(triangle_mesh)
plt.show()