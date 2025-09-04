import numpy as np
import matplotlib.pyplot as plt


def cartesian_to_spherical(x: float, y: float, z: float):
    """
    Converts : x, y, z --> r, fi, lat
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    fi = np.arctan2(y, x)
    lat = np.arctan2(z, np.sqrt(x**2 + y**2))
    return r, fi, lat


def spherical_to_cartesian(r: float, fi: float, lat: float):
    """
    converts : r, fi, lat --> x, y, z
    """
    x = r * np.cos(fi) * np.cos(lat)
    y = r * np.sin(fi) * np.cos(lat)
    z = r * np.sin(lat)
    return x, y, z


def vec_ortho(B: np.ndarray):
    """
    Takes B vector and returns 2 orthogonal vectors u1, u2 (u1 in XZ plane)
    """
    Bx = B[0]
    By = B[1]
    Bz = B[2]
    Bxz = np.sqrt(B[0] ** 2 + B[2] ** 2)
    B = np.linalg.norm(B)
    u1 = np.array([Bz / Bxz, 0, -Bx / Bxz])
    u2 = np.array([By * u1[2] / B, Bxz / B, -By * u1[0] / B])
    return u1, u2


def draw_sphere_3D():
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    u = np.linspace(0, 2 * np.pi, 200)
    v = np.linspace(0, np.pi, 200)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, rstride=4, cstride=4, color="grey", linewidth=0, alpha=1)
    ax.set_xlabel("x ($R_M$)", fontsize=15, fontname="DejaVu Serif")
    ax.set_ylabel("y ($R_M$)", fontsize=15, fontname="DejaVu Serif")
    ax.set_zlabel("z ($R_M$)", fontsize=15, fontname="DejaVu Serif")
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.tick_params(axis="z", labelsize=12)
    ax.set_aspect("equal")
    return fig, ax
