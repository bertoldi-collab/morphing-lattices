import jax.numpy as jnp
from morphing_lattices.kinematics import build_strain_fn
from morphing_lattices.structure import ControlParams


def build_strain_energy(connectivity):

    strain_fn = build_strain_fn(connectivity)

    def strain_energy(displacement: jnp.ndarray, control_params: ControlParams):
        strain = strain_fn(control_params.reference_points, displacement)
        reference_points = control_params.reference_points
        reference_vectors = reference_points[connectivity[:, 1]] - reference_points[connectivity[:, 0]]
        reference_lengths = jnp.linalg.norm(reference_vectors, axis=-1)

        young = control_params.young
        area = control_params.area
        thermal_strain = control_params.thermal_strain

        # NOTE: If we want to include the change of cross-sectional area, we can add it here with a dedicated alpha_transverse.
        # From experiments: alpha_transverse ~ alpha_longitudinal.

        # Linear
        return 0.5 * jnp.sum(
            young * area * reference_lengths * ((1+strain)/(1+thermal_strain) - 1)**2
        )
    
    return strain_energy
