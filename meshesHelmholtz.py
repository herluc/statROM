import gmsh
import numpy as np
import ufl
#import dolfinx.plotting
from dolfinx.fem import Function, FunctionSpace
import dolfinx

from mpi4py import MPI
proc = MPI.COMM_WORLD.rank

from dolfinx.io import XDMFFile, gmshio#,extract_gmsh_geometry,extract_gmsh_topology_and_markers, ufl_mesh_from_gmsh
from dolfinx.cpp.io import perm_gmsh
from dolfinx.cpp.mesh import to_type
from dolfinx.mesh import create_mesh


def generate_mesh_with_obstacle(Lx=1,Ly=1,lc=.03,refine=0.04):
    # For further documentation see
    # - gmsh tutorials, e.g. see https://gitlab.onelab.info/gmsh/gmsh/-/blob/master/tutorial/python/t10.py 
    # - dolfinx-gmsh interface https://github.com/FEniCS/dolfinx/blob/master/python/demo/gmsh/demo_gmsh.py
    # 
    gmsh.initialize()
    gdim = 2
    
    if proc == 0:
        model = gmsh.model()
        model.add("Rectangle minus Circle")
        model.setCurrent("Rectangle minus Circle")
        rect = model.occ.addRectangle(0,0,0,Lx,Ly)#
        circle = model.occ.addDisk(0.5,0.5,0.0,0.08,0.08)
        #circle2 = model.occ.addDisk(0.8,0.7,0.0,0.15,0.15)
        #slit = model.occ.addRectangle(0.5,0.75,0.0,0.001,0.255)
        #box = model.occ.addRectangle(0.30,0.3,0.0,0.1,0.1)
        p1 = gmsh.model.geo.addPoint(0.25, 0.25, 0, lc, 11)
        p2 = gmsh.model.geo.addPoint(0.5, 0.7, 0, lc, 12)

        plate = model.occ.cut([(2,rect)], [(2,circle)])
        #model.occ.cut([(2,rect)], [(2,circle2)])
        #model.occ.cut([(2,rect)], [(2,box)])

        model.occ.synchronize()
        model.geo.synchronize()
        surface_entities_ = model.getEntities(dim=2)
        marker = 11

        #surface_entities = [model[1] for model in model.getEntities(2)]
        #model.addPhysicalGroup(2, surface_entities, tag=5)
        #model.setPhysicalName(2, 2, "Rectangle surface")
        surface_entities = model.getEntities(dim=gdim)
        assert(len(surface_entities) == 1)
        gmsh.model.addPhysicalGroup(surface_entities[0][0], [surface_entities[0][1]], 5)
        gmsh.model.setPhysicalName(surface_entities[0][0], 5, "Rectangle surface")

        bottom_marker, right_marker, wall_marker, obstacle_marker = 20,30,40,50
        bottom, outflow, walls, obstacle = [], [], [], []

        boundaries = model.getBoundary(surface_entities, oriented=False)
        for boundary in boundaries:
            center_of_mass = model.occ.getCenterOfMass(boundary[0], boundary[1])
            if np.allclose(center_of_mass, [Lx/2, Ly, 0]) or np.allclose(center_of_mass, [Lx, Ly/2, 0]) or np.allclose(center_of_mass, [0, Ly/2, 0]):
                walls.append(boundary[1])
            elif np.allclose(center_of_mass, [Lx/2, 0, 0]):
                walls.append(boundary[1])
            else:
                obstacle.append(boundary[1])
                
        model.addPhysicalGroup(1, walls, wall_marker)
        model.setPhysicalName(1, wall_marker, "Walls")
        model.addPhysicalGroup(1, bottom, bottom_marker)
        model.setPhysicalName(1, bottom_marker, "Bottom")
        model.addPhysicalGroup(1, outflow, right_marker)
        model.setPhysicalName(1, right_marker, "Outlet")
        model.addPhysicalGroup(1, obstacle, obstacle_marker)
        model.setPhysicalName(1, obstacle_marker, "Obstacle")

        model.mesh.field.add("Distance", 1)
        model.mesh.field.setNumbers(1, "PointsList", [p1])


        # We then define a `Threshold' field, which uses the return value of the
        # `Distance' field 1 in order to define a simple change in element size
        # depending on the computed distances
        #
        # SizeMax -                     /------------------
        #                              /
        #                             /
        #                            /
        # SizeMin -o----------------/
        #          |                |    |
        #        Point         DistMin  DistMax
        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "InField", 1)
        gmsh.model.mesh.field.setNumber(2, "SizeMin", lc / 10)
        gmsh.model.mesh.field.setNumber(2, "SizeMax", lc/10)
        gmsh.model.mesh.field.setNumber(2, "DistMin", 0.3)
        gmsh.model.mesh.field.setNumber(2, "DistMax", 0.7)

     
        gmsh.model.mesh.field.add("Distance", 4)
        gmsh.model.mesh.field.setNumbers(4, "PointsList", [11])

    
        gmsh.model.mesh.field.add("MathEval", 5)
        gmsh.model.mesh.field.setString(5, "F", "F4^6 + " + str(lc / 10))

        gmsh.model.mesh.field.add("Distance", 8)
        gmsh.model.mesh.field.setNumbers(8, "PointsList", [12])

        gmsh.model.mesh.field.add("MathEval", 9)
        gmsh.model.mesh.field.setString(9, "F", "F8^6 + " + str(lc / 10))

        gmsh.model.mesh.field.add("Min", 7)
        gmsh.model.mesh.field.setNumbers(7, "FieldsList", [2, 5, 9])

        gmsh.model.mesh.field.setAsBackgroundMesh(7)

        # The API also allows to set a global mesh size callback, which is called each
        # time the mesh size is queried
        def meshSizeCallback(dim, tag, x, y, z, lc):
            return min(lc, refine) #0.04 #0.023

        gmsh.model.mesh.setSizeCallback(meshSizeCallback)

        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)


        gmsh.option.setNumber("Mesh.Algorithm", 5)
        model.occ.synchronize()
        model.geo.synchronize()

        model.mesh.generate(gdim)


        # Create a DOLFINx mesh (same mesh on each rank)
        msh, cell_markers, facet_markers = gmshio.model_to_mesh(model, MPI.COMM_SELF, 0)
        msh.name = "Cracked"
        cell_markers.name = f"{msh.name}_cells"
        facet_markers.name = f"{msh.name}_facets"
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "output/elasticity-demo.xdmf", "w") as file:
            file.write_mesh(msh)
        
    
    with dolfinx.io.XDMFFile(msh.comm, "ft.xdmf", "w") as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_meshtags(facet_markers)
    return msh, cell_markers, facet_markers 



