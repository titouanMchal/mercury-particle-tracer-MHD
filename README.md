# mercury-particle-tracer-MHD
This repository provides a Python tool to compute the trajectory of charged particles in Mercury’s magnetosphere, using electromagnetic fields from an MHD model: Block Adaptive Tree Solar wind Roe-type Upwind Scheme (BATSRUS) code.

## Repository structure

- **`tracer.py`**
  - `Ion`: defines an ion species (name, charge [C], mass [kg]).
  - `Shoot`: runs a single trajectory integration 
  - `push()`: convenience function to configure and launch a simulation.

- **`functions.py`**
  - spherical/cartesian transforms, orthogonal basis from B, simple 3D sphere plot.

- **`MHD_model.py`** 
  - Load the 3D grids for **B** and **E** from `fields.npz`.
  - To get the field at an arbitrary position (in meters) : `interpolate(coord, field)` 
  - Includes 2D slice helpers like `slice_B()` / `slice_E()` for visualization.

- **`example_trajectories.py`** (example script)
  - Shows test-trajectories for a sodium ion in the cusp and a solar‐wind proton,
  - Shows 2D/3D plots and physical diagnostics (energy, magnetic moment, field components).

- **`counter.py`**
  - Defines the `Grid3DCounter` class, which discretizes space into a 3D grid of cubic cells.
  - Tracks how many times a trajectory visits each cell.
  - Provides methods to:
    - count and reset visited cells,
    - retrieve cell centers,
    - visualize visited cells in 3D,
    - plot counts on boundary faces or slices

## Getting Started
```python
import tracer as tr
```

First, initialize ion species :

```python
sodium = tr.Ion(name="Na+", charge=1.6e-19, mass=23*1.67252e-27)
```

Then, perform a simulation with `push()`
```python
shoot = tr.push(particle=sodium, x0_RM=np.array(func.spherical_to_cartesian(r=1.01, fi=np.pi, lat=np.deg2rad(70))), kinetic_energy_eV=10, pitch_deg=150, phase_deg=0, dt=1e-2, direction=1)
```
Extract trajectory :
```python
trajectory = shoot.x
```
