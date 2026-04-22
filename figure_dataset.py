import trimesh
import numpy as np
from scipy.spatial.transform import Rotation

SEED = 42
np.random.seed(SEED)

def generate_dataset(number_of_samples, number_of_augmentations):

    shapes = [
        ("Box",      trimesh.creation.box()),
        ("Sphere",   trimesh.creation.icosphere()),
        ("Cone",     trimesh.creation.cone(radius=1, height=2)),
        ("Cylinder", trimesh.creation.cylinder(radius=1, height=2)),
        ("Pyramid",  trimesh.creation.cone(radius=1, height=2, sections=4)),
    ]

    X = []
    y = []

    for i , (name, mesh) in enumerate(shapes):
        points = mesh.sample(number_of_samples)
        for j in range(number_of_augmentations):
            R = Rotation.random().as_matrix()
            points_rotated = points @ R.T
            X.append(points_rotated)
            y.append(i)
            
    return np.asarray(X) , np.asarray(y)


def visualize_dataset(X, y):
    import matplotlib.pyplot as plt

    labels = ["Box", "Sphere", "Cone", "Cylinder", "Pyramid"]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    n_classes = len(labels)

    _, axes = plt.subplots(1, n_classes, figsize=(4 * n_classes, 4),
                             subplot_kw={"projection": "3d"})

    for cls in range(n_classes):
        idx = np.where(y == cls)[0][0]
        pts = X[idx]
        ax = axes[cls]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, c=colors[cls])
        ax.set_title(labels[cls])
        ax.set_axis_off()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    X, y = generate_dataset(number_of_samples=100, number_of_augmentations=1)
    visualize_dataset(X, y)
