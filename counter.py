import numpy as np
import matplotlib.pyplot as plt
import MHD_model
from matplotlib.patches import Circle
from matplotlib.colors import LogNorm

RM = MHD_model.RM


class Grid3DCounter:
    def __init__(self, xmin, xmax, ymin, ymax, zmin, zmax, dx):
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax
        self.zmin, self.zmax = zmin, zmax
        self.dx = dx
        self.nx = int(np.ceil((xmax - xmin) / dx))
        self.ny = int(np.ceil((ymax - ymin) / dx))
        self.nz = int(np.ceil((zmax - zmin) / dx))
        self.counts = np.zeros((self.nx, self.ny, self.nz), dtype=int)

    def _to_indices(self, points):
        """
        Convert (N,3) array in grid indices (i,j,k).
        If point is outside : -1
        """
        shifted = (points - np.array([self.xmin, self.ymin, self.zmin])) / self.dx
        idx = np.floor(shifted).astype(int)
        valid = (
            (idx[:, 0] >= 0)
            & (idx[:, 0] < self.nx)
            & (idx[:, 1] >= 0)
            & (idx[:, 1] < self.ny)
            & (idx[:, 2] >= 0)
            & (idx[:, 2] < self.nz)
        )
        idx[~valid] = -1
        return idx, valid

    def count_trajectory(self, trajectory):
        """
        Increments self.counts for each visited cell
        """
        idx, valid = self._to_indices(trajectory)
        idx_valid = idx[valid]
        visited = set(map(tuple, idx_valid))
        for i, j, k in visited:
            self.counts[i, j, k] += 1
        return visited

    def reset_counts(self):
        self.counts.fill(0)

    def get_cell_centers(self, indices):
        """
        Compute cell centers for a list of cells
        """
        centers = []
        for i, j, k in indices:
            x = self.xmin + (i + 0.5) * self.dx
            y = self.ymin + (j + 0.5) * self.dx
            z = self.zmin + (k + 0.5) * self.dx
            centers.append((x, y, z))
        return np.array(centers)

    def plot_visited_cells(self, trajectory, visited_indices, show_trajectory=True):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        if show_trajectory:
            ax.plot(
                trajectory[:, 0] / RM,
                trajectory[:, 1] / RM,
                trajectory[:, 2] / RM,
                color="blue",
                alpha=0.7,
                linewidth=1,
                label="trajectory",
                zorder=4.5,
            )

        centers = self.get_cell_centers(visited_indices)
        ax.scatter(
            centers[:, 0] / RM,
            centers[:, 1] / RM,
            centers[:, 2] / RM,
            c="red",
            s=20,
            depthshade=True,
            label=f"visited cells ({len(centers)})",
        )
        u = np.linspace(0, 2 * np.pi, 200)
        v = np.linspace(0, np.pi, 200)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(
            x, y, z, rstride=4, cstride=4, color="grey", linewidth=0, alpha=1
        )
        ax.set_xlabel("x ($R_M$)", fontsize=15, fontname="DejaVu Serif")
        ax.set_ylabel("y ($R_M$)", fontsize=15, fontname="DejaVu Serif")
        ax.set_zlabel("z ($R_M$)", fontsize=15, fontname="DejaVu Serif")
        ax.set_aspect("equal")
        ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_face(self, face="xmin", cmap="viridis"):
        """
        Plot a boundary counting matrix
        face : {'xmin','xmax','ymin','ymax','zmin','zmax'}
        cmap : str
        """
        fig, ax = plt.subplots()
        if face == "xmin":
            data = self.counts[0, :, :].T
            extent = [self.ymin / RM, self.ymax / RM, self.zmin / RM, self.zmax / RM]
            xlabel, ylabel = "Y", "Z"
            tit = "x = -3 $R_M$"
        elif face == "xmax":
            data = self.counts[-1, :, :].T
            extent = [self.ymin / RM, self.ymax / RM, self.zmin / RM, self.zmax / RM]
            xlabel, ylabel = "Y", "Z"
            tit = "x = 5 $R_M$"
        elif face == "ymin":
            data = self.counts[:, 0, :].T
            extent = [self.xmin / RM, self.xmax / RM, self.zmin / RM, self.zmax / RM]
            xlabel, ylabel = "X", "Z"
            tit = "y = -6 $R_M$"
        elif face == "ymax":
            data = self.counts[:, -1, :].T
            extent = [self.xmin / RM, self.xmax / RM, self.zmin / RM, self.zmax / RM]
            xlabel, ylabel = "X", "Z"
            tit = "y = 6 $R_M$"
        elif face == "zmin":
            data = self.counts[:, :, 0].T
            extent = [self.xmin / RM, self.xmax / RM, self.ymin / RM, self.ymax / RM]
            xlabel, ylabel = "X", "Y"
            tit = "z = -4 $R_M$"
        elif face == "zmax":
            data = self.counts[:, :, -1].T
            extent = [self.xmin / RM, self.xmax / RM, self.ymin / RM, self.ymax / RM]
            xlabel, ylabel = "X", "Y"
            tit = "z = 4 $R_M$"
        else:
            raise ValueError(
                f"Face '{face}' invalid. Choose xmin,xmax,ymin,ymax,zmin,zmax."
            )

        im = ax.imshow(data, origin="lower", extent=extent, aspect="auto", cmap=cmap)
        cbar = fig.colorbar(im, ax=ax, label="Counts")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(tit)
        return fig, ax

    def plot_slice_z(self, z_value, cmap="viridis"):
        """
        z_value : float
        cmap : str
        """
        fig, ax = plt.subplots()
        z_value = z_value * RM
        k = int(np.floor((z_value - self.zmin) / self.dx))
        if k < 0 or k >= self.nz:
            raise ValueError(
                f"z_value={z_value} outside domain [{self.zmin}, {self.zmax}]."
            )
        data = self.counts[:, :, k].T
        extent = [self.xmin / RM, self.xmax / RM, self.ymin / RM, self.ymax / RM]
        circle = Circle((0, 0), 1, fill=0, color="gray", zorder=4)
        ax.add_patch(circle)

        im = ax.imshow(
            data,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap=cmap,
            norm=LogNorm(),
        )
        cbar = fig.colorbar(im, ax=ax, label="Counts")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"Slice Z={z_value/RM} $R_M$ (k={k})")
        ax.set_aspect("equal")
        return fig, ax
