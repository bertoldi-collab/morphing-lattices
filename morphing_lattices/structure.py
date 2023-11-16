from dataclasses import dataclass
from typing import Dict, NamedTuple, Optional
import jax.numpy as jnp


class ControlParams(NamedTuple):
    """Parameters that define the lattice structure and can be optimized.

    Attrs:
        reference_points (jnp.ndarray): Array of shape (n_points, 2) representing the reference nodal positions of the lattice.
        young (jnp.ndarray): Array of shape (n_bonds,) representing the Young's modulus of each struts.
        area (jnp.ndarray): Array of shape (n_bonds,) representing the cross-sectional area of each struts.
        thermal_strain (jnp.ndarray): Array of shape (n_bonds,) representing the thermal strain of each struts.
        damping (jnp.ndarray): Array of shape (n_points, 2) representing the damping of each point.
        masses (jnp.ndarray): Array of shape (n_points, 2) representing the mass of each point.
        loading_params (Dict): Loading parameters to be passed to loading_fn. Default: {}.
        constraint_params (Dict): Constraint parameters to be passed to constraint_fn. Default: {}.
    """

    reference_points: jnp.ndarray
    young: jnp.ndarray
    area: jnp.ndarray
    thermal_strain: jnp.ndarray
    damping: jnp.ndarray
    masses: jnp.ndarray
    loading_params: Dict = dict()
    constraint_params: Dict = dict()


@dataclass
class Lattice:
    """Lattice structure.

    Attrs:
        connectivity (jnp.ndarray): Array of shape (n_bonds, 2) representing the connectivity of the lattice.
        control_params (ControlParams): Parameters that define the lattice structure and can be optimized.
        solution (jnp.ndarray): Array of shape (n_timepoints, 2, n_points, 2) representing the nodal positions of the lattice. Default: None.
    """

    connectivity: jnp.ndarray
    control_params: ControlParams
    solution: Optional[jnp.ndarray] = None

    # Add post init
    def __post_init__(self):
        self.n_points = self.control_params.reference_points.shape[0]
        self.n_bonds = self.connectivity.shape[0]

    def get_xy_limits(self, extra_x=0.05, extra_y=0.05):
        points = self.control_params.reference_points
        x_ext = points[:, 0].max() - points[:, 0].min()
        y_ext = points[:, 1].max() - points[:, 1].min()
        xlim = (points[:, 0].min() - extra_x*x_ext,
                points[:, 0].max() + extra_x*x_ext)
        ylim = (points[:, 1].min() - extra_y*y_ext,
                points[:, 1].max() + extra_y*y_ext)
        return xlim, ylim
