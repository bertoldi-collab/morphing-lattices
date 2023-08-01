import jax.numpy as jnp
from typing import Callable, Dict, List
from morphing_lattices.geometry import DOFsInfo


def build_loading(
        n_points: int,
        loaded_point_DOF_pairs: jnp.ndarray,
        loading_fn: Callable,
        constrained_point_DOF_pairs: jnp.ndarray = jnp.array([]),
        rigid_bodies_points: List[jnp.ndarray] = [],):
    """Defines the loading function.

    Args:
        n_points (int): Number of points in the geometry.
        loaded_point_DOF_pairs (jnp.ndarray): array of shape (Any, 2) where each row defines a pair of [point_id, DOF_id] where DOF_id is either 0, 1, or 2
        loading_fn (Callable): Loading function. Output shape should either be scalar or match (len(loaded_point_DOF_pairs),).
        constrained_point_DOF_pairs (jnp.ndarray, optional): Array of shape (n_constraints, 2) where each row is of the form [point_id, DOF_id]. Defaults to jnp.array([]).
        rigid_bodies_points (List[jnp.ndarray], optional): List of arrays of shape (n_points_in_rigid_body, ) defining the points in each rigid body. Defaults to [].

    Returns:
        Callable: vector loading function evaluating to `loading_fn` for the DOFs defined by `loaded_point_DOF_pairs` and 0 otherwise.
    """

    # loaded DOF ids based on global numeration
    loaded_DOF_ids = jnp.array([point_id * 2 + DOF_id for point_id, DOF_id in loaded_point_DOF_pairs])
    # Retrieve free DOFs from constraints info (this information is assumed to be static)
    free_DOF_ids, _, _, all_DOF_ids = DOFsInfo(n_points, constrained_point_DOF_pairs, rigid_bodies_points)

    def _loading_fn(state, t, loading_params: Dict):

        loading_vector = jnp.zeros((len(all_DOF_ids),))
        if len(loaded_point_DOF_pairs) != 0:
            loading_vector = loading_vector.at[loaded_DOF_ids].set(
                loading_fn(state, t, **loading_params)
            )
        loading_vector = loading_vector[free_DOF_ids]  # Reduce loading vector to the free DOFs

        return loading_vector

    return _loading_fn


def build_global_loading(
        n_points: int,
        loaded_point_DOF_pairs: jnp.ndarray,
        loading_fn: Callable,):
    """Defines the global loading function (used for post-processing).

    Args:
        n_points (int): Number of points in the geometry.
        loaded_point_DOF_pairs (jnp.ndarray): array of shape (Any, 2) where each row defines a pair of [point_id, DOF_id] where DOF_id is either 0, 1, or 2
        loading_fn (Callable): Loading function. Output shape should either be scalar or match (len(loaded_point_DOF_pairs),).

    Returns:
        Callable: function with signature `loading_fn_global(state, t, loading_params)` -> ndarray of shape (n_points, 2) representing the global loading vector.
    """

    def _loading_fn(state, t, loading_params: Dict):

        loading_vector = jnp.zeros((n_points, 2))
        if len(loaded_point_DOF_pairs) != 0:
            loading_vector = loading_vector.at[loaded_point_DOF_pairs[:, 0], loaded_point_DOF_pairs[:, 1]].set(
                loading_fn(state, t, **loading_params)
            )

        return loading_vector

    return _loading_fn
