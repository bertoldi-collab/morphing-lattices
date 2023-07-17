from dataclasses import dataclass
from typing import Dict, NamedTuple, Optional
import jax.numpy as jnp


class ControlParams(NamedTuple):
    reference_points: jnp.ndarray
    young: jnp.ndarray  # Stiffness in the reference configuration. This will be a function of temperature.
    area: jnp.ndarray  # Cross-sectional areas in the reference configuration.
    thermal_strain: jnp.ndarray  # Thermal strain. This will be a function of temperature.
    damping: jnp.ndarray  # Damping coefficient.
    masses: jnp.ndarray  # Masses of the points.
    loading_params: Dict = dict()  # Loading parameters to be passed to loading functions. Default: {}.
    constraint_params: Dict = dict()  # Constraint parameters to be passed to constraint_DOFs_fn. Default: {}.


@dataclass
class Lattice:
    connectivity: jnp.ndarray
    control_params: ControlParams
    solution: Optional[jnp.ndarray] = None

    # Add post init
    def __post_init__(self):
        self.n_points = self.control_params.reference_points.shape[0]
        self.n_bonds = self.connectivity.shape[0]
