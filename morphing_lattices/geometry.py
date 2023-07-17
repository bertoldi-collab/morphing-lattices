import jax.numpy as jnp
from typing import List


# Utility functions


def DOFsInfo(n_points: int, constrained_point_DOF_pairs: jnp.ndarray, rigid_bodies_points: List[jnp.ndarray] = []):
    """Computes arrays defining the free, constrained, and all DOFs.

    Args:
        n_points (int): Number of points in the geometry
        constrained_point_DOF_pairs (jnp.ndarray, optional): Array of shape (n_constraints, 2) where each row is of the form [point_id, DOF_id]. Defaults to jnp.array([]).
        rigid_bodies_points (List[jnp.ndarray], optional): List of arrays of shape (n_points_in_rigid_body, ) defining the points in each rigid body. Defaults to [].

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: arrays defining the free, constrained, and all DOFs.
    """

    constrained_DOF_ids = jnp.array([block_id * 2 + DOF_id for block_id, DOF_id in constrained_point_DOF_pairs])
    all_DOF_ids = jnp.arange(n_points * 2)
    free_DOF_ids = jnp.array([dof for dof in all_DOF_ids if dof not in constrained_DOF_ids])

    # Handle rigid body constraints
    if len(rigid_bodies_points) > 0:
        rigid_body_DOF_ids = [jnp.concatenate([(point_id*2 + jnp.arange(2)) for point_id in rigid_body_points])
                              for rigid_body_points in rigid_bodies_points]  # All DOFs of the points in rigid bodies
        # First 3 IDs of the first rigid body are considered as reference (x, y and theta)
        rigid_body_DOF_dependent_ids = jnp.concatenate([DOF_ids[3:] for DOF_ids in rigid_body_DOF_ids])
        # Remove the dependent DOFs from the free DOFs
        free_DOF_ids = jnp.array([dof for dof in free_DOF_ids if dof not in rigid_body_DOF_dependent_ids])
    else:
        rigid_body_DOF_ids = []

    return free_DOF_ids, constrained_DOF_ids, rigid_body_DOF_ids, all_DOF_ids


def points_centroid(points: jnp.ndarray, masses: jnp.ndarray = jnp.array(1.)):
    """Computes the centroid of a set of points.

    Args:
        points (jnp.ndarray): array of shape (n, 2)
        masses (jnp.ndarray, optional): array of shape (n,). Defaults to jnp.array(1.).

    Returns:
        jnp.ndarray: array of shape (2,)
    """

    return ((points.T*masses).T.sum(axis=0)/masses.sum())


def rigid_bodies_inertia(points: jnp.ndarray, masses: jnp.ndarray):
    """Computes the inertia of a set of points moving as one rigid body.

    Args:
        centroids (jnp.ndarray): array of shape (n, 2)
        masses (jnp.ndarray): array of shape (n, 2) of masses.

    Returns:
        jnp.ndarray: inertia of the rigid body
    """

    centroid = points_centroid(points, masses=masses[:, 0])
    mass = jnp.sum(masses[:, 0])

    return jnp.array([
        mass,
        mass,
        jnp.sum(masses[:, 0]*((points - centroid)**2).T)
    ])


def compute_inertia(points: jnp.ndarray, masses: jnp.ndarray, rigid_bodies_points: List[jnp.ndarray] = [],):
    """Computes inertia of the system by taking the masses and modifying the inertia of the rigid bodies.

    Args:
        points (jnp.ndarray): array of shape (n, 2)
        masses (jnp.ndarray): array of shape (n, 2) of masses.
        rigid_bodies_points (List[jnp.ndarray], optional): List of arrays of shape (n_points_in_rigid_body, ) defining the points in each rigid body. Defaults to [].

    Returns:
        jnp.ndarray: array of shape (2*n_points,) collecting the translational and rotational inertia of the system.
    """

    inertia = masses.reshape(-1)

    for rigid_body_points in rigid_bodies_points:
        inertia = inertia.at[rigid_body_points[0]*2:rigid_body_points[0]*2+3].set(
            rigid_bodies_inertia(
                points[rigid_body_points],
                masses[rigid_body_points]
            )
        )

    return inertia


# Triangular lattice


def triangular_lattice_points(n1: int, n2: int, spacing: float):
    """Generates the reference points of a triangular lattice.

    Args:
        n1 (int): Number of points in the first direction.
        n2 (int): Number of points in the second direction.
        spacing (float): Spacing between points.

    Returns:
        jnp.ndarray: array of shape (n_points, 2) collecting the points of the lattice.
    """

    row = jnp.array([jnp.arange(n1+1), jnp.zeros(n1+1)]).T * spacing
    return jnp.concatenate(
        [
            row[:n1+1-jnp.mod(i, 2)] + jnp.array([
                jnp.mod(i, 2)*jnp.cos(jnp.pi/3),
                i*jnp.sin(jnp.pi/3)
            ])*spacing
            for i in range(n2+1)]
    )


def triangular_lattice_connectivity(n1: int, n2: int):
    """Generates the connectivity of a triangular lattice.

    Args:
        n1 (int): Number of points in the first direction.
        n2 (int): Number of points in the second direction.

    Returns:
        jnp.ndarray: array of shape (n_bonds, 2) collecting the connectivity of the lattice.
    """

    horizontal_bonds_even = jnp.concatenate(
        [jnp.array([jnp.arange(n1)+i*(2*n1+1), jnp.arange(n1)+i*(2*n1+1)+1]).T for i in range(n2//2+1)])
    horizontal_bonds_odd = jnp.concatenate(
        [jnp.array([jnp.arange(n1-1)+(i+1)*(n1+1)+i*n1, jnp.arange(n1-1)+(i+1)*(n1+1)+i*n1+1]).T for i in range(n2//2)])

    right_leaning_even = jnp.concatenate(
        [jnp.array([jnp.arange(n1), jnp.arange(n1)+n1+1]).T + i*(2*n1+1) for i in range(n2//2)])
    right_leaning_odd = jnp.concatenate(
        [jnp.array([jnp.arange(n1), jnp.arange(n1)+n1+1]).T + i*(2*n1+1) + n1+1 for i in range(n2//2)])

    left_leaning_even = jnp.concatenate(
        [jnp.array([jnp.arange(n1)+1, jnp.arange(n1)+1+n1]).T + i*(2*n1+1) for i in range(n2//2)])
    left_leaning_odd = jnp.concatenate(
        [jnp.array([jnp.arange(n1)+1, jnp.arange(n1)+1+n1]).T + i*(2*n1+1) + n1 for i in range(n2//2)])

    connectivity = jnp.concatenate([horizontal_bonds_even, horizontal_bonds_odd,
                                    right_leaning_even, right_leaning_odd, left_leaning_even, left_leaning_odd])
    return connectivity
