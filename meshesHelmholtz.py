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




def generate_mesh_with_hole(Lx=1,Ly=1,Lcrack=.3,lc=.015,refinement_ratio=10,dist_min=.05,dist_max=.2):
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
        circle = model.occ.addDisk(0.35,0.2,0.0,0.3,0.025)
        circle2 = model.occ.addDisk(0.35,0.5,0.0,0.3,0.025)
        #slit = model.occ.addRectangle(0.5,0.75,0.0,0.001,0.255)
        box = model.occ.addRectangle(0.1,0.2,0.0,0.03,0.3)
        p1 = gmsh.model.geo.addPoint(0.25, 0.25, 0, lc, 11)
        p2 = gmsh.model.geo.addPoint(0.5, 0.7, 0, lc, 12)

        plate = model.occ.cut([(2,rect)], [(2,circle)])
        model.occ.cut([(2,rect)], [(2,circle2)])
        model.occ.cut([(2,rect)], [(2,box)])

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
        gmsh.model.mesh.field.setNumber(2, "SizeMin", lc / 30)
        gmsh.model.mesh.field.setNumber(2, "SizeMax", lc/30)
        gmsh.model.mesh.field.setNumber(2, "DistMin", 0.3)
        gmsh.model.mesh.field.setNumber(2, "DistMax", 0.7)

        # # Say we want to modulate the mesh element sizes using a mathematical function
        # # of the spatial coordinates. We can do this with the MathEval field:
        # gmsh.model.mesh.field.add("MathEval", 3)
        # gmsh.model.mesh.field.setString(3, "F",
        #                                 "Cos(4*3.14*x) * Sin(4*3.14*y) / 10 + 0.101")

        # We could also combine MathEval with values coming from other fields. For
        # example, let's define a `Distance' field around point 1
        gmsh.model.mesh.field.add("Distance", 4)
        gmsh.model.mesh.field.setNumbers(4, "PointsList", [11])

        # We can then create a `MathEval' field with a function that depends on the
        # return value of the `Distance' field 4, i.e., depending on the distance to
        # point 1 (here using a cubic law, with minimum element size = lc / 100)
        gmsh.model.mesh.field.add("MathEval", 5)
        gmsh.model.mesh.field.setString(5, "F", "F4^6 + " + str(lc / 30))


        gmsh.model.mesh.field.add("Distance", 8)
        gmsh.model.mesh.field.setNumbers(8, "PointsList", [12])

        gmsh.model.mesh.field.add("MathEval", 9)
        gmsh.model.mesh.field.setString(9, "F", "F8^6 + " + str(lc / 30))

        # # We could also use a `Box' field to impose a step change in element sizes
        # # inside a box
        # gmsh.model.mesh.field.add("Box", 6)
        # gmsh.model.mesh.field.setNumber(6, "VIn", lc / 15)
        # gmsh.model.mesh.field.setNumber(6, "VOut", lc)
        # gmsh.model.mesh.field.setNumber(6, "XMin", 0.3)
        # gmsh.model.mesh.field.setNumber(6, "XMax", 0.6)
        # gmsh.model.mesh.field.setNumber(6, "YMin", 0.3)
        # gmsh.model.mesh.field.setNumber(6, "YMax", 0.6)
        # gmsh.model.mesh.field.setNumber(6, "Thickness", 0.3)

        # Many other types of fields are available: see the reference manual for a
        # complete list. You can also create fields directly in the graphical user
        # interface by selecting `Define->Size fields' in the `Mesh' module.

        # Let's use the minimum of all the fields as the mesh size field:
        gmsh.model.mesh.field.add("Min", 7)
        gmsh.model.mesh.field.setNumbers(7, "FieldsList", [2, 5, 9])

        gmsh.model.mesh.field.setAsBackgroundMesh(7)

        # The API also allows to set a global mesh size callback, which is called each
        # time the mesh size is queried
        def meshSizeCallback(dim, tag, x, y, z, lc):
            return min(lc, 0.02 * x + 0.01)

        gmsh.model.mesh.setSizeCallback(meshSizeCallback)

        # To determine the size of mesh elements, Gmsh locally computes the minimum of
        #
        # 1) the size of the model bounding box;
        # 2) if `Mesh.MeshSizeFromPoints' is set, the mesh size specified at geometrical
        #    points;
        # 3) if `Mesh.MeshSizeFromCurvature' is positive, the mesh size based on
        #    curvature (the value specifying the number of elements per 2 * pi rad);
        # 4) the background mesh size field;
        # 5) any per-entity mesh size constraint;
        #
        # The value can then be further modified by the mesh size callback, if any,
        # before being constrained in the interval [`Mesh.MeshSizeMin',
        # `Mesh.MeshSizeMax'] and multiplied by `Mesh.MeshSizeFactor'.  In addition,
        # boundary mesh sizes are interpolated inside surfaces and/or volumes depending
        # on the value of `Mesh.MeshSizeExtendFromBoundary' (which is set by default).
        #
        # When the element size is fully specified by a mesh size field (as it is in
        # this example), it is thus often desirable to set

        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

        # This will prevent over-refinement due to small mesh sizes on the boundary.

        # Finally, while the default "Frontal-Delaunay" 2D meshing algorithm
        # (Mesh.Algorithm = 6) usually leads to the highest quality meshes, the
        # "Delaunay" algorithm (Mesh.Algorithm = 5) will handle complex mesh size fields
        # better - in particular size fields with large element size gradients:

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
        
    
    #     # Sort mesh nodes according to their index in gmsh
    #     geometry_data = extract_gmsh_geometry(model, model_name="Rectangle")[:,0:2]
    #     topology_data = extract_gmsh_topology_and_markers(model, "Rectangle")
        
    #     # Broadcast cell type data and geometric dimension
    #     gmsh_cell_id = MPI.COMM_WORLD.bcast(model.mesh.getElementType("triangle", 1), root=0)
    #     # Extract the cell type and number of nodes per cell and broadcast
    #     # it to the other processors 
    #     gmsh_cell_type = list(topology_data.keys())[0]    
    #     properties = gmsh.model.mesh.getElementProperties(gmsh_cell_type)
    #     name, dim, order, num_nodes, local_coords, _ = properties
    #     cells = topology_data[gmsh_cell_type]["topology"]
    #     cell_id, num_nodes = MPI.COMM_WORLD.bcast([gmsh_cell_type, num_nodes], root=0)
    # else:        
    #     cell_id, num_nodes = MPI.COMM_WORLD.bcast([None, None], root=0)
    #     cells, geometry_data = np.empty([0, num_nodes]), np.empty([0, gdim])

    # #Permute topology data from MSH-ordering to dolfinx-ordering
    # ufl_domain = ufl_mesh_from_gmsh(cell_id, gdim)
    # gmsh_cell_perm = perm_gmsh(to_type(str(ufl_domain.ufl_cell())), num_nodes)
    # cells = cells[:, gmsh_cell_perm]
    
    # # Create distributed mesh
    # mesh = create_mesh(MPI.COMM_WORLD, cells, geometry_data[:, :gdim], ufl_domain)
    with dolfinx.io.XDMFFile(msh.comm, "ft.xdmf", "w") as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_meshtags(facet_markers)
    return msh, cell_markers, facet_markers 




def generate_mesh_with_crack(Lx=1,Ly=1,Lcrack=.3,lc=.015,refinement_ratio=10,dist_min=.05,dist_max=.2):
    # For further documentation see
    # - gmsh tutorials, e.g. see https://gitlab.onelab.info/gmsh/gmsh/-/blob/master/tutorial/python/t10.py 
    # - dolfinx-gmsh interface https://github.com/FEniCS/dolfinx/blob/master/python/demo/gmsh/demo_gmsh.py
    # 
    gmsh.initialize()
    gdim = 2
    
    if proc == 0:
        model = gmsh.model()
        model.add("Rectangle")
        model.setCurrent("Rectangle")
        p1 = model.geo.addPoint(0.0, 0.0, lc)
        p2 = model.geo.addPoint(Lcrack, 0.0, lc)
        p3 = model.geo.addPoint(Lx, 0, lc)
        p4 = model.geo.addPoint(Lx, Ly, lc)
        p5 = model.geo.addPoint(0, Ly, lc)
        l1 = model.geo.addLine(p1, p2)
        l2 = model.geo.addLine(p2, p3)
        l3 = model.geo.addLine(p3, p4)
        l4 = model.geo.addLine(p4, p5)
        l5 = model.geo.addLine(p5, p1)
        cloop1 = model.geo.addCurveLoop([l1, l2, l3, l4, l5])
        surface_1 = model.geo.addPlaneSurface([cloop1])
        
        model.mesh.field.add("Distance", 1)
        model.mesh.field.setNumbers(1, "NodesList", [p2])
        #model.mesh.field.setNumber(1, "NNodesByEdge", 100)
        #model.mesh.field.setNumbers(1, "EdgesList", [2])
        #
        # SizeMax -                     /------------------
        #                              /
        #                             /
        #                            /
        # SizeMin -o----------------/
        #          |                |    |
        #        Point         DistMin  DistMax
    
        model.mesh.field.add("Threshold", 2)
        model.mesh.field.setNumber(2, "IField", 1)
        model.mesh.field.setNumber(2, "LcMin", lc / refinement_ratio)
        model.mesh.field.setNumber(2, "LcMax", lc)
        model.mesh.field.setNumber(2, "DistMin", dist_min)
        model.mesh.field.setNumber(2, "DistMax", dist_max)
        model.mesh.field.setAsBackgroundMesh(2)
    
    
        model.geo.synchronize()
        surface_entities = [model[1] for model in model.getEntities(2)]
        model.addPhysicalGroup(2, surface_entities, tag=5)
        model.setPhysicalName(2, 2, "Rectangle surface")
        model.mesh.generate(gdim)


        # Create a DOLFINx mesh (same mesh on each rank)
        msh, cell_markers, facet_markers = gmshio.model_to_mesh(model, MPI.COMM_SELF, 0)
        msh.name = "Cracked"
        cell_markers.name = f"{msh.name}_cells"
        facet_markers.name = f"{msh.name}_facets"
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "output/elasticity-demo.xdmf", "w") as file:
            file.write_mesh(msh)
        
    
    #     # Sort mesh nodes according to their index in gmsh
    #     geometry_data = extract_gmsh_geometry(model, model_name="Rectangle")[:,0:2]
    #     topology_data = extract_gmsh_topology_and_markers(model, "Rectangle")
        
    #     # Broadcast cell type data and geometric dimension
    #     gmsh_cell_id = MPI.COMM_WORLD.bcast(model.mesh.getElementType("triangle", 1), root=0)
    #     # Extract the cell type and number of nodes per cell and broadcast
    #     # it to the other processors 
    #     gmsh_cell_type = list(topology_data.keys())[0]    
    #     properties = gmsh.model.mesh.getElementProperties(gmsh_cell_type)
    #     name, dim, order, num_nodes, local_coords, _ = properties
    #     cells = topology_data[gmsh_cell_type]["topology"]
    #     cell_id, num_nodes = MPI.COMM_WORLD.bcast([gmsh_cell_type, num_nodes], root=0)
    # else:        
    #     cell_id, num_nodes = MPI.COMM_WORLD.bcast([None, None], root=0)
    #     cells, geometry_data = np.empty([0, num_nodes]), np.empty([0, gdim])

    # #Permute topology data from MSH-ordering to dolfinx-ordering
    # ufl_domain = ufl_mesh_from_gmsh(cell_id, gdim)
    # gmsh_cell_perm = perm_gmsh(to_type(str(ufl_domain.ufl_cell())), num_nodes)
    # cells = cells[:, gmsh_cell_perm]
    
    # # Create distributed mesh
    # mesh = create_mesh(MPI.COMM_WORLD, cells, geometry_data[:, :gdim], ufl_domain)
    return msh

if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    from pathlib import Path

    Lcrack = 0.5
    dist_min = .1
    dist_max = .3
    mesh = generate_mesh_with_crack(Lcrack=Lcrack,
                     Ly=.5,
                     lc=.1, # caracteristic length of the mesh
                     refinement_ratio=10, # how much it is refined at the tip zone
                     dist_min=dist_min, # radius of tip zone
                     dist_max=dist_max # radius of the transition zone 
                     )

    Path("output").mkdir(parents=True, exist_ok=True)    
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "output/elasticity-demo.xdmf", "w") as file:
        file.write_mesh(mesh)
