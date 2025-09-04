import tracer as tr
import matplotlib.pyplot as plt
import numpy as np
import functions as func
import MHD_model

"""
This is an example script showing how to compute and visualize test-particle trajectories 
in the electromagnetic fields derived from an MHD model of Mercury's environment. 

Two ion species are initialized:
    - A sodium ion (Na+) injected near the cusp region with low energy
    - A proton injected from the solar wind with higher energy

For each particle, the trajectory is integrated using Boris method. 
Several types of diagnostics and visualizations are provided :

    * 2D trajectory plots (XZ and XY planes),
    * 3D trajectory plot,
    * physical diagnostics such as:
        - kinetic energy vs time,
        - magnetic moment vs time,
        - magnetic and electric field components along the trajectory.

The script can be used as a template for exploring how different initial 
conditions (species, energy, pitch angle, location) affect particle dynamics 
inside Mercury's magnetosphere.
"""

RM = MHD_model.RM

sodium = tr.Ion(name="Na+", charge=1.6e-19, mass=23 * 1.67252e-27)
proton = tr.Ion(name="proton", charge=1.6e-19, mass=1.672649e-27)


plot_2D = True
plot_3D = False
plot_physical_diagnostic = True


shoot_cusp = tr.push(
    particle=sodium,
    x0_RM=np.array(func.spherical_to_cartesian(r=1.01, fi=np.pi, lat=np.deg2rad(70))),
    kinetic_energy_eV=10,
    pitch_deg=150,
    phase_deg=0,
    dt=1e-2,
    direction=1,
)
shoot_sw = tr.push(
    particle=proton,
    x0_RM=np.array([-3, 0.2, 1]),
    kinetic_energy_eV=650,
    pitch_deg=10,
    phase_deg=0,
    dt=1e-2,
    direction=1,
)


if plot_2D:
    fig1, ax1 = MHD_model.slice_B(projection="XZ", stream=1, y_RM=0)
    fig2, ax2 = MHD_model.slice_B(projection="XY", stream=0)

    ax1.plot(
        shoot_cusp.x[:, 0] / RM,
        shoot_cusp.x[:, 2] / RM,
        color="blue",
        linewidth=1,
        zorder=4,
    )
    ax2.plot(
        shoot_cusp.x[:, 0] / RM,
        shoot_cusp.x[:, 1] / RM,
        color="blue",
        linewidth=1,
        zorder=4,
    )
    ax1.plot(
        shoot_sw.x[:, 0] / RM,
        shoot_sw.x[:, 2] / RM,
        color="darkred",
        linewidth=1,
        zorder=4,
    )
    ax2.plot(
        shoot_sw.x[:, 0] / RM,
        shoot_sw.x[:, 1] / RM,
        color="darkred",
        linewidth=1,
        zorder=4,
    )

if plot_3D:
    fig, ax = func.draw_sphere_3D()
    ax.plot(
        shoot_cusp.x[:, 0] / RM,
        shoot_cusp.x[:, 1] / RM,
        shoot_cusp.x[:, 2] / RM,
        zorder=4,
        linewidth=0.5,
        color="blue",
    )
    ax.plot(
        shoot_sw.x[:, 0] / RM,
        shoot_sw.x[:, 1] / RM,
        shoot_sw.x[:, 2] / RM,
        zorder=4,
        linewidth=0.5,
        color="darkred",
    )
    ax.set_aspect("equal")


if plot_physical_diagnostic:

    # kinetic energy vs time
    fig, ax = plt.subplots(figsize=(8, 6))
    shoot_cusp.plot_kinetic_energy(ax=ax, color="blue", label="sodium, cusp")
    shoot_sw.plot_kinetic_energy(ax=ax, color="darkred", label="proton, SW, cusp")

    # magnetic moment vs time
    fig, ax = plt.subplots(figsize=(8, 8))
    shoot_cusp.plot_mu(ax=ax, color="blue")
    shoot_sw.plot_mu(ax=ax, color="darkred")

    # magnetic field components vs time
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 5))
    shoot_cusp.plot_B(axes=axes, color="blue")
    shoot_sw.plot_B(axes=axes, color="darkred")

    # electric field components vs time
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
    shoot_cusp.plot_E(axes=axes, color="blue")
    shoot_sw.plot_E(axes=axes, color="darkred")


plt.show()
