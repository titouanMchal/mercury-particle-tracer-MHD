import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import LogNorm


RM = 2440e3  # Mercury's radius in m

# You need to provide a 3D grid in .npz format.
# In this example, the grids are derived from MHD simulation results obtained using the Block Adaptive Tree Solar wind Roe-type Upwind Scheme (BATSRUS) code.
# SW velocity = [350, -50, 0] km/s
# IMF = [40, -20, -15] nT
# SW density = 150 cm^-3
# SW temperature = 7.5 eV
# Adjust the domain boundaries and cell size accordingly if you change the 3D grids.
# B in nT
# E in mV/m

fields = np.load("fields.npz")

B = fields["B"] * 1e-9
E = fields["E"] * 1e-3

I, J, K, _ = B.shape

DX = 0.04  # cell size in RM
X_MIN = -3
X_MAX = 5
Y_MIN = -6
Y_MAX = 6
Z_MIN = -4
Z_MAX = 4


def conversion_simu_to_grid(coord: np.ndarray):
    """
    Inputs
    ------
    coord
        coordinates in the simulation frame

    Returns
    -------
    coordinates in the grid frame
    """
    c = coord / RM
    x, y, z = c[0], c[1], c[2]
    x_grid, y_grid, z_grid = x - X_MIN, y - Y_MIN, z - Z_MIN
    return np.array([x_grid, y_grid, z_grid])


def find_cell(coord: np.ndarray):
    """
    Takes a position vector and finds the corresponding cell in the grid

    Inputs
    ------
    coord
        coordinates in the simulation frame

    Returns
    -------
    i, j, k, ri, rj, rk
    """
    x_grid = conversion_simu_to_grid(coord)
    i, j, k = int(x_grid[0] // DX), int(x_grid[1] // DX), int(x_grid[2] // DX)
    ri, rj, rk = x_grid[0] % DX, x_grid[1] % DX, x_grid[2] % DX
    return i, j, k, ri, rj, rk


def interpolate(coord: np.ndarray, field: np.ndarray):
    """
    Manual trilinear interpolation (looks faster than scipy.ndimage map_coordinates)

    Inputs
    ------
    coord
        coordinates in the simulation frame
    field
        3D array storing E or B values

    Returns
    -------
    Interpolated field
    """
    i, j, k, ri, rj, rk = find_cell(coord)
    ri, rj, rk = ri / DX, rj / DX, rk / DX
    f000 = field[i, j, k]
    f100 = field[i + 1, j, k]
    f010 = field[i, j + 1, k]
    f110 = field[i + 1, j + 1, k]
    f001 = field[i, j, k + 1]
    f101 = field[i + 1, j, k + 1]
    f011 = field[i, j + 1, k + 1]
    f111 = field[i + 1, j + 1, k + 1]

    f00 = f000 * (1 - ri) + f100 * ri
    f10 = f010 * (1 - ri) + f110 * ri
    f01 = f001 * (1 - ri) + f101 * ri
    f11 = f011 * (1 - ri) + f111 * ri

    f0 = f00 * (1 - rj) + f10 * rj
    f1 = f01 * (1 - rj) + f11 * rj

    f_interp = f0 * (1 - rk) + f1 * rk

    return f_interp


def slice_B(
    projection: str = "XZ",
    y_RM: float = 0,
    z_RM: float = 0,
    planet: bool = True,
    fill_planet: bool = True,
    stream: bool = False,
    color_magnitude: bool = True,
):
    fig, ax = plt.subplots()
    ax.set_xlabel("X ($R_M$)", fontsize=13, fontname="DejaVu Serif")

    if projection == "XZ":
        j = int((y_RM - Y_MIN) // DX)
        slice = np.linalg.norm(B[:, j, :].T * 1e9, axis=0)
        vmax = 1000
        ax.set_ylim(Z_MIN, Z_MAX)
        ax.set_xlim(X_MIN, X_MAX)
        ax.set_ylabel("Z ($R_M$)", fontsize=13, fontname="DejaVu Serif")
        if color_magnitude:
            im = ax.imshow(
                slice,
                origin="lower",
                extent=[X_MIN, X_MAX, Z_MIN, Z_MAX],
                interpolation="bilinear",
                norm=LogNorm(vmax=vmax),
            )

    if projection == "XY":
        k = int((z_RM - Z_MIN) // DX)
        slice = np.linalg.norm(B[:, :, k].T * 1e9, axis=0)
        vmax = 800
        ax.set_ylim(Y_MIN, Y_MAX)
        ax.set_xlim(X_MIN, X_MAX)
        ax.set_ylabel("Y ($R_M$)", fontsize=13, fontname="DejaVu Serif")
        if color_magnitude:
            im = ax.imshow(
                slice,
                origin="lower",
                extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
                interpolation="bilinear",
                norm=LogNorm(vmax=vmax),
            )

    if planet:
        if fill_planet:
            ax.add_patch(Circle((0, 0), 1, fill=1, color="gray", zorder=4))
            angle_start = 90
            angle_end = 270
            angles = np.linspace(np.radians(angle_start), np.radians(angle_end), 100)
            x = np.cos(angles)
            y = np.sin(angles)
            ax.fill_betweenx(y, x, 0, color="lightgray", zorder=4)
        else:
            ax.add_patch(Circle((0, 0), 1, fill=0, color="black", zorder=4))

    if stream:
        if projection == "XZ":
            x_vals = np.linspace(X_MIN, X_MIN + DX * (I - 1), I)  # de −5 à +3 R_M
            z_vals = np.linspace(Z_MIN, Z_MIN + DX * (K - 1), K)  # de −4 à +4 R_M
            Xg, Zg = np.meshgrid(x_vals, z_vals)  # Xg, Zg shape = (K, I)
            Bx_slice = B[:, j, :, 0].T * 1e9  # transpose → (K, I)
            Bz_slice = B[:, j, :, 2].T * 1e9  # transpose → (K, I)
            stream = ax.streamplot(
                Xg,
                Zg,
                Bx_slice,
                Bz_slice,
                density=0.5,
                color="white",
                linewidth=0.1,
                arrowsize=0.8,
                integration_direction="both",
                broken_streamlines=0,
            )

        if projection == "XY":
            x_vals = np.linspace(X_MIN, X_MIN + DX * (I - 1), I)
            y_vals = np.linspace(Y_MIN, Y_MIN + DX * (J - 1), J)
            Xg, Yg = np.meshgrid(x_vals, y_vals)
            Bx_slice = B[:, :, k, 0].T
            By_slice = B[:, :, k, 1].T
            stream = ax.streamplot(
                Xg,
                Yg,
                Bx_slice,
                By_slice,
                density=0.3,
                color="white",
                linewidth=0.1,
                arrowsize=0.8,
                integration_direction="both",
                broken_streamlines=0,
            )
    if color_magnitude:
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(
            r"$\Vert \vec{B} \Vert$ (nT)", fontsize=13, fontname="DejaVu Serif"
        )
    ax.set_aspect("equal")
    return fig, ax


def slice_E(
    projection: str = "XZ",
    j: int = 150,
    k: int = 100,
    planet: bool = True,
    fill_planet: bool = True,
    stream: bool = False,
):
    fig, ax = plt.subplots()
    ax.set_xlabel("X ($R_M$)", fontsize=13, fontname="DejaVu Serif")

    if projection == "XZ":
        slice = np.linalg.norm(E[:, j, :].T * 1e3, axis=0)
        ax.set_ylim(Z_MIN, Z_MAX)
        ax.set_xlim(X_MIN, X_MAX)
        ax.set_ylabel("Z ($R_M$)", fontsize=13, fontname="DejaVu Serif")
        im = ax.imshow(
            slice,
            origin="lower",
            extent=[X_MIN, X_MAX, Z_MIN, Z_MAX],
            interpolation="bilinear",
        )

    if projection == "XY":
        slice = np.linalg.norm(E[:, :, k].T * 1e9, axis=0)
        vmax = 800
        ax.set_ylim(Y_MIN, Y_MAX)
        ax.set_xlim(X_MIN, X_MAX)
        ax.set_ylabel("Y ($R_M$)", fontsize=13, fontname="DejaVu Serif")
        im = ax.imshow(
            slice,
            origin="lower",
            extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
            interpolation="bilinear",
        )

    if planet:
        if fill_planet:
            ax.add_patch(Circle((0, 0), 1, fill=1, color="gray", zorder=4))
            angle_start = 90
            angle_end = 270
            angles = np.linspace(np.radians(angle_start), np.radians(angle_end), 100)
            x = np.cos(angles)
            y = np.sin(angles)
            ax.fill_betweenx(y, x, 0, color="lightgray", zorder=4)
        else:
            ax.add_patch(Circle((0, 0), 1, fill=0, color="black", zorder=4))

    if stream:
        if projection == "XZ":
            x_vals = np.linspace(X_MIN, X_MIN + DX * (I - 1), I)
            z_vals = np.linspace(Z_MIN, Z_MIN + DX * (K - 1), K)
            Xg, Zg = np.meshgrid(x_vals, z_vals)  # Xg, Zg shape = (K, I)
            Ex_slice = E[:, j, :, 0].T  # transpose → (K, I)
            Ez_slice = E[:, j, :, 2].T  # transpose → (K, I)
            stream = ax.streamplot(
                Xg,
                Zg,
                Ex_slice,
                Ez_slice,
                density=0.5,
                color="white",
                linewidth=0.1,
                arrowsize=0.8,
            )

        if projection == "XY":
            x_vals = np.linspace(X_MIN, X_MIN + DX * (I - 1), I)
            y_vals = np.linspace(Y_MIN, Y_MIN + DX * (J - 1), J)
            Xg, Yg = np.meshgrid(x_vals, y_vals)
            Ex_slice = E[:, :, k, 0].T
            Ey_slice = E[:, :, k, 1].T
            stream = ax.streamplot(
                Xg,
                Yg,
                Ex_slice,
                Ey_slice,
                density=0.5,
                color="white",
                linewidth=0.1,
                arrowsize=0.6,
                integration_direction="both",
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(
        r"$\Vert \vec{E} \Vert$ (mV/m)", fontsize=13, fontname="DejaVu Serif"
    )
    return fig, ax


if __name__ == "__main__":

    slice_B(projection="XZ", stream=True)
    slice_B(projection="XY", stream=True)
    plt.show()
