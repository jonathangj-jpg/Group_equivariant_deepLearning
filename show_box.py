import trimesh
import numpy as np

N = 10

shapes = [
    ("Box",      trimesh.creation.box()),
    ("Sphere",   trimesh.creation.icosphere()),
    ("Cone",     trimesh.creation.cone(radius=1, height=2)),
    ("Cylinder", trimesh.creation.cylinder(radius=1, height=2)),
    ("Pyramid",  trimesh.creation.cone(radius=1, height=2, sections=4)),
]

sphere_bottom = shapes[1][1].bounds[0][2]  # min z af kuglen

scene = trimesh.Scene()

for i, (name, mesh) in enumerate(shapes):
    # Flugter bunden med kuglens bund
    z_offset = sphere_bottom - mesh.bounds[0][2]
    offset = np.array([i * 3, 0, z_offset])
    mesh.apply_translation(offset)

    mesh.visual.face_colors = [180, 180, 180, 255]
    scene.add_geometry(mesh, node_name=name)

    points = mesh.sample(N)
    for j, pt in enumerate(points):
        dot = trimesh.creation.icosphere(radius=0.03)
        dot.apply_translation(pt)
        dot.visual.face_colors = [255, 0, 0, 255]
        scene.add_geometry(dot, node_name=f"{name}_sample_{j}")

scene.show()
