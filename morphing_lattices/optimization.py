from dataclasses import dataclass
import dataclasses
from typing import Any, Callable, List, Optional
from jax import jit, value_and_grad, flatten_util
import jax.numpy as jnp
import nlopt
from morphing_lattices.dynamics import setup_dynamic_solver

from morphing_lattices.structure import ControlParams, Lattice


def material_interpolation(phase, value_1, value_2, p=3):
    return value_1 + (value_2-value_1)*phase**p


@dataclass
class ForwardProblem:
    """
    Forward problem for SIMP-based optimization of shape morphing LCE lattices.

    Attrs:
        lattice (Lattice): Lattice structure
        simulation_time (float): Simulation time
        n_timepoints (int): Number of timepoints
        is_setup (Optional[bool]): Flag indicating that solve method is not available. It needs to be set up by calling self.setup(). Default: False.
        name (str): Problem name. Default: "shape_morphing".
    """

    # Lattice structure
    lattice: Lattice

    # Analysis params
    simulation_time: Any
    n_timepoints: int

    # Flag indicating that solve method is not available. It needs to be set up by calling self.setup().
    is_setup: bool = False

    # Problem name
    name: str = "shape_morphing"

    def setup(self, young_1_fn: Callable, young_2_fn: Callable, thermal_strain_1_fn: Callable, thermal_strain_2_fn: Callable):
        """
        Set up forward solver.
        """

        # Initial conditions
        state0 = jnp.zeros((2, self.lattice.n_points, 2))

        # Analysis params
        timepoints = jnp.linspace(0, self.simulation_time, self.n_timepoints)

        # Material properties evolution
        def control_params_fn(t, control_params: ControlParams):
            return control_params._replace(
                young=material_interpolation(
                    phase=control_params.phase,
                    value_1=young_1_fn(t),
                    value_2=young_2_fn(t),
                ),
                thermal_strain=material_interpolation(
                    phase=control_params.phase,
                    value_1=thermal_strain_1_fn(t),
                    value_2=thermal_strain_2_fn(t),
                )
            )

        # Setup solver
        solve_dynamics, _ = setup_dynamic_solver(
            lattice=self.lattice,
            control_params_fn=control_params_fn,
        )

        # Setup forward fn
        def forward(phase: jnp.ndarray):

            # Define control params for the current design
            control_params = self.lattice.control_params._replace(
                phase=phase,
            )

            # Solve dynamics
            solution = solve_dynamics(
                state0=state0,
                timepoints=timepoints,
                control_params=control_params,
            )

            return solution, control_params

        self.solve = forward
        self.control_params_fn = control_params_fn
        self.is_setup = True

    @staticmethod
    def from_data(problem_data):
        problem_data = ForwardProblem(**problem_data)
        problem_data.is_setup = False
        return problem_data

    def to_data(self):
        return ForwardProblem(**dataclasses.asdict(self))


@dataclass
class OptimizationProblem:
    """
    Optimization problem for SIMP-based optimization of shape morphing LCE lattices.

    Attrs:
        forward_problem (ForwardProblem): Forward problem
        target_points (jnp.ndarray): Array of shape (n_target_points, 2) representing the target points.
        target_points_ids (jnp.ndarray): Array of shape (n_target_points,) representing the target points ids.
        objective_values (Optional[List[Any]]): List of objective values. Default: None.
        design_values (Optional[List[Any]]): List of design values. Default: None.
        best_response (Optional[jnp.ndarray]): Array of shape (n_timepoints, 2, n_points, 2) representing the displacement and velocity of each point at each timepoint for the best design. Default: None.
        best_control_params (Optional[ControlParams]): Parameters that define the best lattice structure. Default: None.
        name (str): Problem name. Default: ForwardProblem.name.
        is_setup (Optional[bool]): Flag indicating that objective_fn method is not available. It needs to be set up by calling self.setup_objective(). Default: False.
    """

    forward_problem: ForwardProblem
    target_points: jnp.ndarray
    target_points_ids: jnp.ndarray
    objective_values: Optional[List[Any]] = None
    design_values: Optional[List[Any]] = None
    best_response: Optional[jnp.ndarray] = None
    best_control_params: Optional[ControlParams] = None
    name: str = ForwardProblem.name

    # Flag indicating that objective_fn method is not available. It needs to be set up by calling self.setup_objective().
    is_setup: bool = False

    def __post_init__(self):
        self.objective_values = [] if self.objective_values is None else self.objective_values
        self.design_values = [] if self.design_values is None else self.design_values

    def setup_objective(self) -> None:
        """
        Jit compiles the objective function.
        """

        # Make sure forward solvers are set up
        assert self.forward_problem.is_setup, "Forward problem is not set up. Call self.forward_problem.setup() first."

        def distance_from_target_shape(phase: jnp.ndarray):

            # Solve forward
            solution, control_params = self.forward_problem.solve(phase)
            # TODO: Implement distance from target shape
            # TODO: Think about how to specify two target shapes
            # NOTE: Do we want to try a scale invariant distance? I think we need it actually.
            # NOTE: As a test, we can try to minimize the distance from the final configuration
            final_configuration = control_params.reference_points + \
                solution[-1, 0]

            return jnp.sum((final_configuration[self.target_points_ids] - self.target_points)**2)

        self.objective_fn = distance_from_target_shape
        self.is_setup = True

    def run_optimization_nlopt(
            self,
            n_iterations: int,
            max_time: Optional[int] = None,
            lower_bound=0.,
            upper_bound=1.,):

        # TODO: Figure out wether or not we need filering

        # Make sure objective_fn is set up
        if not self.is_setup:
            self.setup_objective()

        initial_guess = self.forward_problem.lattice.control_params.phase

        def flatten(tree): return flatten_util.ravel_pytree(tree)[0]
        _, unflatten = flatten_util.ravel_pytree(initial_guess)

        objective_and_grad = jit(value_and_grad(self.objective_fn))

        def nlopt_objective(x, grad):

            v, g = objective_and_grad(unflatten(x))  # jax evaluation
            self.objective_values.append(v)
            self.design_values.append(unflatten(x))

            print(
                f"Iteration: {len(self.objective_values)}\nObjective = {self.objective_values[-1]}")

            if grad.size > 0:
                grad[:] = flatten(g)

            return float(v)

        initial_guess_flattened = flatten(initial_guess)
        opt = nlopt.opt(nlopt.LD_MMA, len(initial_guess_flattened))
        opt.set_param("verbosity", 1)
        opt.set_maxeval(n_iterations)

        opt.set_min_objective(nlopt_objective)
        opt.set_lower_bounds(lower_bound)
        opt.set_upper_bounds(upper_bound)

        if max_time is not None:
            opt.set_maxtime(max_time)

        # Run optimization
        opt.optimize(initial_guess_flattened)

        # Store forward solution data for the last design
        self.compute_best_response()

    def compute_best_response(self):

        if len(self.design_values) == 0:
            raise ValueError("No design has been optimized yet.")

        # Make sure forward solvers are set up
        assert self.forward_problem.is_setup, "Forward problem is not set up. Call self.forward_problem.setup() first."

        self.best_response, self.best_control_params = self.forward_problem.solve(
            self.design_values[-1]
        )

        return self.best_response, self.best_control_params

    @staticmethod
    def from_data(optimization_data):
        optimization_data.forward_problem = ForwardProblem.from_data(
            optimization_data.forward_problem
        )
        optimization_data.is_setup = False
        return optimization_data

    def to_data(self):
        return OptimizationProblem(**dataclasses.asdict(self))
