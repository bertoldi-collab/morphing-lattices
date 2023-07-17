from typing import Callable, List
import jax.numpy as jnp
from morphing_lattices.geometry import DOFsInfo, points_centroid
from morphing_lattices.structure import ControlParams


def rotation_matrix(angle):
    return jnp.array([[jnp.cos(angle), -jnp.sin(angle)],
                      [jnp.sin(angle), jnp.cos(angle)]])


def rigid_body_displacement(DOFs: jnp.ndarray, centroid_points: jnp.ndarray):
    """Computes the displacement of a rigid body from the DOFs of its points.

    Args:
        DOFs (jnp.ndarray): Array of shape (3, 2) containing the DOFs of the rigid body.
        centroid_points (jnp.ndarray): Array of shape (n_points, 2) containing the points of the rigid body in the centroid frame.

    Returns:
        jnp.ndarray: Array of shape (n_points, 2) containing the displacements of the points in the rigid body.
    """

    # Compute the displacement of all the points in the rigid body
    centroid_displacement = DOFs[:2]
    centroid_rotation = DOFs[2]
    point_displacement = centroid_displacement + \
        jnp.dot(rotation_matrix(centroid_rotation) - jnp.eye(2), centroid_points.T).T

    return point_displacement


def build_constrained_kinematics(n_points: int, constrained_point_DOF_pairs: jnp.ndarray, rigid_bodies_points: List[jnp.ndarray] = [], constrained_DOFs_fn: Callable = lambda t, **kwargs: 0):
    """Defines a constrained kinematics of the points.

    Args:
        n_points (int): Number of points in the geometry
        constrained_point_DOF_pairs (jnp.ndarray): Array of shape (n_constraints, 2) where each row is of the form [point_id, DOF_id].
        rigid_bodies_points (List[jnp.ndarray], optional): List of arrays of shape (n_points_in_rigid_body, ) defining the points in each rigid body. Defaults to [].
        constrained_DOFs_fn (Callable, optional): Constraint function defining how the DOFs are driven over time. Output shape should either be scalar or match (len(constrained_point_DOF_pairs),). Valid signature: `constrained_DOFs_fn(t, **kwargs) -> ndarray`. Defaults to lambda t: 0.

    Returns:
        Callable: Constraint function mapping the free DOFs and time to the displacement of all the points. The signature is `constrained_kinematics(free_DOFs, t, constraint_params)`.
    """

    # Retrieve free DOFs from constraints info (this information is assumed to be static)
    free_DOF_ids, constrained_DOF_ids, rigid_body_DOF_ids, all_DOF_ids = DOFsInfo(
        n_points, constrained_point_DOF_pairs, rigid_bodies_points
    )

    def constrained_kinematics(free_DOFs: jnp.ndarray, t, control_params: ControlParams):
        """Constrained kinematics of the blocks.

        Args:
            free_DOFs (jnp.ndarray): Array of shape (n_free_DOFs,) representing the free DOFs.
            t (float): Time parameter for time-dependent constraints.
            control_params (ControlParams): ControlParams.

        Returns:
            jnp.ndarray: Array of shape (n_points, 2) representing the DOFs of all the points.
        """

        all_DOFs = jnp.zeros((len(all_DOF_ids),))
        # Assign imposed displacements along the constrained DOFs
        if len(constrained_DOF_ids) != 0:
            all_DOFs = all_DOFs.at[constrained_DOF_ids].set(
                constrained_DOFs_fn(t, **control_params.constraint_params)
            )
        # Simply assign the free_DOFs along the free DOFs (this acts as the identity operator)
        all_DOFs = all_DOFs.at[free_DOF_ids].set(
            free_DOFs
        )
        # Handle rigid body constraints
        if len(rigid_body_DOF_ids) > 0:
            for rigid_ids, point_ids in zip(rigid_body_DOF_ids, rigid_bodies_points):
                # Retrieve the points of the rigid body in the centroid frame
                centroid_points = control_params.reference_points[point_ids] - points_centroid(
                    control_params.reference_points[point_ids],
                    control_params.masses[point_ids, 0]
                )
                # Compute the displacement of the rigid body points
                all_DOFs = all_DOFs.at[rigid_ids].set(
                    rigid_body_displacement(all_DOFs[rigid_ids[:3]], centroid_points).reshape(-1)
                )

        return all_DOFs.reshape((n_points, 2))

    return constrained_kinematics


def constrain_energy(energy_fn: Callable, constrained_kinematics: Callable):
    """Defines a constrained version of `energy_fn` according to `constrained_kinematics`.

    Args:
        energy_fn (Callable): Energy functional to be constrained.
        constrained_kinematics (Callable): Constraint function mapping the free DOFs and time to the displacement of all the points. Normally, this is the output of `build_constrained_kinematics`.

    Returns:
        Callable: Constrained energy functional with signature (free_dofs, time, control_params) -> energy.
    """

    def constrained_energy_fn(free_DOFs: jnp.ndarray, t, control_params: ControlParams):
        return energy_fn(
            constrained_kinematics(free_DOFs, t, control_params),
            control_params
        )

    return constrained_energy_fn


def build_strain_fn(connectivity: jnp.ndarray):

    def strain_fn(reference_points: jnp.ndarray, displacement: jnp.ndarray):
        current_points = reference_points + displacement
        reference_vectors = reference_points[connectivity[:, 1]] - reference_points[connectivity[:, 0]]
        current_vectors = current_points[connectivity[:, 1]] - current_points[connectivity[:, 0]]
        reference_lengths = jnp.linalg.norm(reference_vectors, axis=-1)
        current_lengths = jnp.linalg.norm(current_vectors, axis=-1)
        strain = (current_lengths - reference_lengths) / reference_lengths
        return strain

    return strain_fn
