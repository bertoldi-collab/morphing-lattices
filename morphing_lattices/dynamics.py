import jax.numpy as jnp
from jax import grad, vmap, jacobian
from jax.experimental.ode import odeint
from typing import Callable, List
from morphing_lattices.loading import build_loading
from morphing_lattices.structure import ControlParams, Lattice
from morphing_lattices.energy import build_strain_energy
from morphing_lattices.geometry import DOFsInfo, compute_inertia
from morphing_lattices.kinematics import build_constrained_kinematics, constrain_energy


def setup_dynamic_solver(
        lattice: Lattice,
        loaded_point_DOF_pairs: jnp.ndarray = jnp.array([]),
        loading_fn:  Callable = lambda state, t: 0,
        constrained_point_DOF_pairs: jnp.ndarray = jnp.array([]),
        rigid_bodies_points: List[jnp.ndarray] = [],
        constrained_DOFs_fn: Callable = lambda t, **kwargs: 0,
        control_params_fn: Callable = lambda t, control_params: control_params,):
    """Sets up the dynamic solver for a lattice.

    Args:
        lattice (Lattice): Lattice object.
        loaded_point_DOF_pairs (jnp.ndarray, optional): Array of shape (Any, 2) collecting the point-DOF pairs where the loading is applied. Defaults to jnp.array([]).
        loading_fn (Callable, optional): Function of signature (state, t) -> loading. Defaults to lambda state, t: 0. It should return either a scalar or an array of shape (len(loaded_point_DOF_pairs),).
        constrained_point_DOF_pairs (jnp.ndarray, optional): Array of shape (Any, 2) collecting the point-DOF pairs where the constraints are applied. Defaults to jnp.array([]).
        rigid_bodies_points (List[jnp.ndarray], optional): List of arrays of shape (n_points_in_rigid_body, ) defining the points in each rigid body. Defaults to [].
        constrained_DOFs_fn (Callable, optional): Function of signature (t, **kwargs) -> constrained_DOFs. Defaults to lambda t, **kwargs: 0. It should return either a scalar or an array of shape (len(constrained_point_DOF_pairs),).
        control_params_fn (Callable, optional): Function of signature (t, control_params) -> control_params. Defaults to lambda t, control_params: control_params. It should return a ControlParams object. This is useful to implement time-dependent control parameters.

    Returns:
        Callable, Callable: solve_dynamics, rhs. The first is a function of signature (state0, timepoints, control_params) -> solution. The second is a function of signature (state, t, control_params) -> state_dot. The solution is an array of shape (len(timepoints), 2, n_points, 2), where axis 0 is time, axis 1 is state (displacement, velocity), axis 2 is point id, axis 3 is DOF.
    """

    # Energy
    total_energy = build_strain_energy(lattice.connectivity)
    # NOTE: Handle constraints
    constrained_kinematics = build_constrained_kinematics(
        n_points=lattice.n_points,
        constrained_point_DOF_pairs=constrained_point_DOF_pairs,
        rigid_bodies_points=rigid_bodies_points,
        constrained_DOFs_fn=constrained_DOFs_fn,
    )
    constrained_energy = constrain_energy(energy_fn=total_energy, constrained_kinematics=constrained_kinematics)
    # NOTE: Handle loading
    loading = build_loading(
        n_points=lattice.n_points,
        loaded_point_DOF_pairs=loaded_point_DOF_pairs,
        loading_fn=loading_fn,
        constrained_point_DOF_pairs=constrained_point_DOF_pairs,
        rigid_bodies_points=rigid_bodies_points,
    )

    # Right hand side of the ODE
    potential_force = grad(lambda x, *args, **kwargs: -constrained_energy(x, *args, **kwargs))

    # Retrieve free DOFs from constraints info (this information is assumed to be static)
    free_DOF_ids, _, _, all_DOF_ids = DOFsInfo(lattice.n_points, constrained_point_DOF_pairs, rigid_bodies_points)
    n_all_DOFs = len(all_DOF_ids)

    # Utility functions to reconstruct the full state array from the solution of the free DOFs
    kinematics_history_fn = vmap(constrained_kinematics, in_axes=(0, 0, None))
    jac_kinematics = jacobian(constrained_kinematics, argnums=(0, 1))

    def velocity_fn(free_DOFs, free_DOFs_dot, t, control_params: ControlParams):
        du_dfree, du_dt = jac_kinematics(free_DOFs, t, control_params)
        return du_dfree @ free_DOFs_dot + du_dt

    velocity_history_fn = vmap(velocity_fn, in_axes=(0, 0, 0, None))

    def rhs(state, t, control_params: ControlParams, inertia: jnp.ndarray):
        _control_params = control_params_fn(t, control_params)
        x, x_dot = state
        return jnp.array([
            x_dot,
            (potential_force(x, t, _control_params) + loading(state, t, _control_params.loading_params) -
             _control_params.damping*x_dot) / inertia
        ])

    def solve_dynamics(state0, timepoints, control_params: ControlParams):
        """Solves the dynamics via `odeint`.

        Args:
            state0(jnp.ndarray): array of shape(2, n_points, 2) representing the initial conditions.
            timepoints(jnp.ndarray): evaluation times.
            control_params(ControlParams): control parameters.

        Returns:
            ndarray: Solution of the dynamics evaluated at times `timepoints`. Shape(n_timepoints, 2, n_points, 2), axis 0 is time, axis 1 is state(displacement, velocity), axis 2 is point id, axis 3 is DOF.
        """

        # Reduce state0, masses, and damping to the free DOFs
        _state0 = state0.reshape((2, n_all_DOFs))[:, free_DOF_ids]
        inertia = compute_inertia(
            control_params.reference_points,
            control_params.masses,
            rigid_bodies_points=rigid_bodies_points,
        )[free_DOF_ids]
        _control_params = control_params._replace(damping=control_params.damping.reshape((n_all_DOFs,))[free_DOF_ids])

        # Solve ODE
        free_DOFs_solution = odeint(rhs, _state0, timepoints, _control_params, inertia)

        # Reshape solution to global state.
        kinematic_history = kinematics_history_fn(
            free_DOFs_solution[:, 0, :],
            timepoints,
            control_params
        )
        velocity_history = velocity_history_fn(
            free_DOFs_solution[:, 0, :],
            free_DOFs_solution[:, 1, :],
            timepoints,
            control_params
        )
        solution = jnp.zeros((len(timepoints), 2, lattice.n_points, 2))
        solution = solution.at[:, 0, :, :].set(kinematic_history)
        solution = solution.at[:, 1, :, :].set(velocity_history)

        return solution

    return solve_dynamics, rhs
