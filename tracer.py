import numpy as np
import matplotlib.pyplot as plt
import sys
import MHD_model
import functions as func

plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "font.size": 12,
        "axes.labelsize": 14,
        "grid.alpha": 0.3,
    }
)

RM = MHD_model.RM


def cross(a, b):
    return np.array(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - b[0] * a[1],
        ]
    )


class Ion:
    def __init__(self, name, charge, mass):
        """
        Instantiates an ion species

        Inputs
        ------
        name
            name of the ion (proton for example)
        charge
            electric charge in C
        mass (kg)
        """
        self.name = name
        self.charge = charge
        self.mass = mass


class Shoot:
    def __init__(self, ion, N: int, x0: np.ndarray, v0: np.ndarray, dt: float):
        """
        Instantiates a Shoot object (ion + trajectory + physical diagnostics)

        Inputs
        ------
        ion
            Ion object
        N
            Number of iterations
        x0
            Initial position (m)
        v0
            Initial velocity (m/s)
        dt
            Time step (s) if not adaptive
        """
        self.ion = ion
        self.N = N
        self.dt = dt
        self.t = np.zeros(N)
        self.x = np.zeros((N, 3))
        self.v = np.zeros((N, 3))
        self.x[0] = x0
        self.v[0] = v0
        self.x2 = np.zeros((N, 3))
        self.x3 = np.zeros((N, 3))
        self.e1 = 0
        self.e2 = 0
        self.Tc = np.zeros(N)
        self.Ldt = np.zeros(N)
        self.mu = np.zeros(N)
        self.pitch = np.zeros(N)
        self.B = np.zeros((N, 3))
        self.E = np.zeros((N, 3))
        self.scalar = np.zeros(N)
        self.E_para = np.zeros(N)
        self.E_ortho = np.zeros(N)

    def _boris_step(self, i, B0, E0, direction):
        """
        Computes one iteration with Boris method

        Inputs
        ------
        i : iteration number
        B0 : B field vector
        E0 : E field vector
        direction : +1 / -1  (forward, backward)
        """
        self.v[i] = (
            self.v[i - 1]
            + direction * 0.5 * self.ion.charge * E0 / self.ion.mass * self.dt
        )
        t = self.ion.charge * self.dt * B0 / 2 / self.ion.mass
        v1 = self.v[i] + direction * cross(self.v[i], t)
        self.v[i] = self.v[i] + direction * cross(
            v1, 2 / (1 + np.linalg.norm(t) ** 2) * t
        )
        self.v[i] = (
            self.v[i]
            + direction * self.dt * 1 / 2 * self.ion.charge * E0 / self.ion.mass
        )
        self.x[i] = self.x[i - 1] + direction * self.dt * self.v[i]

    def compute_trajectory_boris(self, precision=360, direction=1, adaptive_dt=False):
        """
        Initialize and run a Boris integrator for an ion in Mercury's magnetosphere.
        """

        for i in range(1, self.N):

            self.B[i - 1] = MHD_model.interpolate(
                coord=self.x[i - 1], field=MHD_model.B
            )  
            nB0 = np.linalg.norm(self.B[i - 1])

            self.E[i - 1] = MHD_model.interpolate(
                coord=self.x[i - 1], field=MHD_model.E
            )  

            if adaptive_dt:
                dt = 2 * np.pi * self.ion.mass / self.ion.charge / nB0 / precision
                if dt < 0.1:
                    self.dt = dt

            self.t[i] = self.t[i - 1] + direction * self.dt
            self.Ldt[i - 1] = self.dt

            self._boris_step(i, B0=self.B[i - 1], E0=self.E[i - 1], direction=direction)

            # mu calculation
            v_drift = np.cross(self.E[i - 1], self.B[i - 1]) / nB0**2
            u1, u2 = func.vec_ortho(self.B[i - 1])
            ec_ortho = (
                1
                / 2
                * self.ion.mass
                * (
                    np.dot(self.v[i - 1] - v_drift, u1) ** 2
                    + np.dot(self.v[i - 1] - v_drift, u2) ** 2
                )
            )
            self.mu[i - 1] = ec_ortho / nB0

            self.Tc[i - 1] = 2 * np.pi * self.ion.mass / self.ion.charge / nB0
            self.E_para[i - 1] = np.dot(self.E[i - 1], self.B[i - 1]) / nB0
            self.E_ortho[i - 1] = np.sqrt(
                np.linalg.norm(self.E[i - 1]) ** 2 - self.E_para[i - 1] ** 2
            )
            self.scalar[i - 1] = (
                np.dot(self.v[i - 1], self.B[i - 1])
                / nB0
                / np.linalg.norm(self.v[i - 1])
            )
            self.pitch[i - 1] = np.arccos(self.scalar[i - 1])

            if np.linalg.norm(self.x[i]) < RM:
                print("Simulation stopped: r < RM")
                break

            if self.x[i][0] >= 5 * RM:
                print("Simulation stopped: x > 5 RM ")
                break

            if self.x[i][0] <= -3 * RM:
                print("\nSimulation stopped: x < -3 RM ")
                break

            if np.abs(self.x[i][1]) >= 6 * RM:
                print("\nSimulation stopped: |y| > 6 RM ")
                break

            if np.abs(self.x[i][2]) >= 4 * RM:
                print("\nSimulation stopped: |z| > 4 RM ")
                break

        index = np.any(self.x != 0, axis=1)
        self.x = self.x[index]
        self.t = self.t[index]
        self.v = self.v[index]
        self.Tc = self.Tc[index]
        self.Ldt = self.Ldt[index]
        self.mu = self.mu[index]
        self.B = self.B[index]
        self.E = self.E[index]
        self.scalar = self.scalar[index]
        self.pitch = self.pitch[index]
        self.E_ortho = self.E_ortho[index]
        self.E_para = self.E_para[index]

    def _prepare_ax(self, ax, xlabel, ylabel, title=None, logy=False):
        if ax is None:
            fig, ax = plt.subplots()
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)
        ax.grid(True)
        return ax

    def plot_trajectory(self, ax=None, color="red", label=None, projection="XZ", w=1):
        if projection == "XZ":
            ax.plot(
                self.x[:, 0] / RM,
                self.x[:, 2] / RM,
                color=color,
                zorder=4,
                label=label,
                linewidth=w,
            )
        if projection == "XY":
            ax.plot(
                self.x[:, 0] / RM,
                self.x[:, 1] / RM,
                color=color,
                zorder=4,
                label=label,
                linewidth=w,
            )
        if label is not None:
            ax.legend()
        return ax

    def plot_kinetic_energy(self, ax=None, color="red", label=None, w=1, title=None):
        t_min = self.t / 60
        ec_keV = (
            0.5 * self.ion.mass * np.linalg.norm(self.v, axis=1) ** 2 / 1.6e-19 / 1e3
        )
        ax = self._prepare_ax(ax, "t (min)", "T (keV)", logy=True, title=title)
        ax.plot(t_min, ec_keV, color=color, label=label, linewidth=w)
        if label is not None:
            ax.legend()
        return ax

    def plot_mu(self, ax=None, color="red", label=None, w=1, title=None):
        t_min = self.t[:-1] / 60
        mu_rel = self.mu[:-1] / self.mu[0]
        ax = self._prepare_ax(ax, "t (min)", r"$\mu/\mu_0$", logy=True, title=title)
        ax.plot(t_min, mu_rel, color=color, label=label, linewidth=w)
        if label is not None:
            ax.legend()
        return ax

    def plot_cyclotron_period(self, ax=None, color="red", label=None, w=1, title=None):
        t = self.t[:-1] / 60
        Tc = self.Tc[:-1]
        ax = self._prepare_ax(ax, "t (min)", "Tc (s)", title=title)
        ax.plot(t, Tc, color=color, label=label, linewidth=w)
        if label:
            ax.legend()
        return ax

    def plot_dt(self, ax=None, color="red", label=None, w=1, title=None):
        t = self.t[:-1] / 60
        dt = self.list_dt[:-1]
        ax = self._prepare_ax(ax, "t (min)", "dt (s)")
        ax.plot(t, dt, color=color, label=label, linewidth=w, title=title)
        if label:
            ax.legend()
        return ax

    def plot_B(self, axes=None, color="red", w=1.2):
        t = self.t[:-1] / 60
        data = self.B[:-1] * 1e9
        if axes is None:
            fig, axes = plt.subplots(3, 1, sharex=True, figsize=(6, 8))
        else:
            fig = axes[0].get_figure()
        labels = ["Bx (nT)", "By (nT)", "Bz (nT)"]
        for i, ax in enumerate(axes):
            ax.plot(t, data[:, i], color=color, linewidth=w)
            ax.set_ylabel(labels[i])
            ax.grid(True)
        axes[-1].set_xlabel("t (min)")
        return fig, axes

    def plot_E(self, axes=None, color="red", w=1.3):
        t = self.t[:-1] / 60
        data = self.E[:-1] * 1e3
        if axes is None:
            fig, axes = plt.subplots(3, 1, sharex=True, figsize=(6, 8))
        else:
            fig = axes[0].get_figure()
        labels = ["Ex (mV/m)", "Ey (mV/m)", "Ez (mV/m)"]
        for i, ax in enumerate(axes):
            ax.plot(t, data[:, i], color=color, linewidth=w)
            ax.set_ylabel(labels[i])
            ax.grid(True)
        axes[-1].set_xlabel("t (min)")
        return fig, axes


def push(
    particle: Ion,
    x0_RM: np.ndarray,
    kinetic_energy_eV: float,
    pitch_deg: float,
    phase_deg: float = 0,
    dt: float = 1e-3,
    n_iter: int = 2000000,
    adaptive_dt: bool = False,
    direction: int = 1,
    precision: int = 360,
) -> Shoot:
    """
    Initialize and run a Boris integrator for an ion in Mercury's magnetosphere.

    Parameters
    ----------
    particle
        An Ion instance (with attributes: name, charge [C], mass [kg]).
    x0_RM
        Initial position in Mercury radii (RM).
    kinetic_energy_eV
        Initial kinetic energy in electron-volts.
    pitch_deg
        Pitch angle in degrees (angle between velocity and magnetic field).

    Other Parameters
    ----------------
    phase_deg : float, optional
        Initial gyrophase in degrees. Default is 0.
    E_field : bool, optional
        Whether to include the electric field. Default is False.
    dse : float, optional
        Step size (in meters) used when computing the electric field. Default is 5e4.
    dt : float, optional
        Fixed time step in seconds (ignored if adaptive_dt=True). Default is 1e-3.
    n_iter : int, optional
        Maximum number of integration steps. Default is 2,000,000.
    phi_drop_V : float, optional
        Potential drop parameter in volts. Default is 20e3.
    adaptive_dt : bool, optional
        If True, the time step is adjusted each iteration based on the cyclotron period. Default is False. Should not be used here.
    direction : int, optional
        +1 to integrate forward in time, -1 to integrate backward. Default is +1.
    energy_corrector : bool, optional
        If True, applies an energy correction at each step. Default is False.
    precision : int, optional
        If adaptive_dt is True, then dt = Tc / precision

    Returns
    -------
    Shoot
        A Shoot object containing the time history of position, velocity, and derived quantities.
    """

    x0 = x0_RM * RM
    phase = np.deg2rad(phase_deg)
    pitch = np.deg2rad(pitch_deg)
    kinetic_energy = kinetic_energy_eV * 1.6e-19
    b0 = MHD_model.interpolate(coord=x0, field=MHD_model.B)
    u1, u2 = func.vec_ortho(b0)
    vpar = np.sqrt(2 * kinetic_energy / particle.mass) * np.cos(pitch)
    vper = np.sqrt(2 * kinetic_energy / particle.mass) * np.sin(pitch)
    v1 = -vper * np.sin(phase)
    v2 = -vper * np.cos(phase)
    v0 = v1 * u1 + v2 * u2 + vpar * b0 / np.linalg.norm(b0)

    shoot = Shoot(ion=particle, N=n_iter, x0=x0, v0=v0, dt=dt)

    shoot.compute_trajectory_boris(
        precision=precision, direction=direction, adaptive_dt=adaptive_dt
    )

    return shoot
