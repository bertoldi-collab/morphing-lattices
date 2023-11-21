#!/usr/bin/env python
# coding: utf-8

# # Inverse design of bimaterial LCE lattice: square
#

# ## Imports

# In[1]:


import multiprocessing
from morphing_lattices.optimization import ForwardProblem, OptimizationProblem
from morphing_lattices.structure import Lattice, ControlParams
from morphing_lattices.geometry import triangular_lattice_points, triangular_lattice_connectivity
from morphing_lattices.plotting import plot_lattice, generate_animation
from morphing_lattices.utils import save_data, load_data
import pandas as pd
import numpy as np
import jax.numpy as jnp
from jax._src.config import config
config.update("jax_enable_x64", True)  # enable float64 type


# In[2]:

def run_optimization(weights: jnp.ndarray):

# ## Import experimental material params

    exp_actuation_strain = pd.read_csv(
        'exp/mechanical_data/20230512_HTNI_LTNI_actuationstrains_highgranularityTemp3355.csv'
    ).drop(['Unnamed: 3'], axis=1)
    exp_modulus_fine = pd.read_csv(
        'exp/mechanical_data/youngs_mod_LCEstrips_DataForBertoldiGroup_updated202307.csv'
    )
    exp_modulus_fine.columns = [
        'Temp',
        'LTNI_avg_young',
        'LTNI_stdev',
        'HTNI_avg_young',
        'HTNI_stdev',
    ]
    exp_modulus_fine.drop([0, 1], inplace=True)
    exp_modulus_fine = exp_modulus_fine.reset_index(drop=True)

    HTNI_stretch_data = exp_actuation_strain[
        ['temp [C]', 'AVG HTNI', 'STD HTNI']
    ].astype(float)
    LTNI_stretch_data = exp_actuation_strain[
        ['temp [C]', 'AVG LTNI', 'STD LTNI']
    ].astype(float)
    HTNI_modulus_data_fine = exp_modulus_fine[
        ['Temp', 'HTNI_avg_young']
    ].astype(float)
    LTNI_modulus_data_fine = exp_modulus_fine[
        ['Temp', 'LTNI_avg_young']
    ].astype(float)

    def HTNI_stretch(temperature):
        return jnp.interp(temperature, jnp.array(HTNI_stretch_data['temp [C]']), jnp.array(HTNI_stretch_data['AVG HTNI']))

    def LTNI_stretch(temperature):
        return jnp.interp(temperature, jnp.array(LTNI_stretch_data['temp [C]']), jnp.array(LTNI_stretch_data['AVG LTNI']))

    def HTNI_young_fit_fine(temperature):
        return jnp.interp(temperature, jnp.array(HTNI_modulus_data_fine['Temp']), jnp.array(HTNI_modulus_data_fine['HTNI_avg_young']))

    def LTNI_young_fit_fine(temperature):
        return jnp.interp(temperature, jnp.array(LTNI_modulus_data_fine['Temp']), jnp.array(LTNI_modulus_data_fine['LTNI_avg_young']))

    # ## Optimization problem

    # ### Temperature evolution

    # In[3]:

    sampled_temperatures = jnp.array([
        22, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130
    ])

    simulation_time = 9000.  # s
    sampled_times = jnp.linspace(
        0, simulation_time, sampled_temperatures.shape[0])

    def temperature_fn(t):
        return jnp.interp(t, sampled_times, sampled_temperatures)

    def young_1_fn(t):
        return HTNI_young_fit_fine(temperature_fn(t))

    def young_2_fn(t):
        return LTNI_young_fit_fine(temperature_fn(t))

    def thermal_strain_1_fn(t):
        return (HTNI_stretch(temperature_fn(t))-1.)*0.8

    def thermal_strain_2_fn(t):
        return (LTNI_stretch(temperature_fn(t))-1.)*0.7

    # ### Forward problem

    # In[4]:

    n1 = 30
    n2 = 30
    spacing = 2.9  # mm
    points = triangular_lattice_points(n1=n1, n2=n2, spacing=spacing)
    connectivity, horiz_bonds_even, horiz_bonds_odd, right_lean_even, right_lean_odd, left_lean_even, left_lean_odd = triangular_lattice_connectivity(
        n1=n1,
        n2=n2
    )

    n_points = points.shape[0]
    n_bonds = connectivity.shape[0]

    control_params = ControlParams(
        reference_points=points,  # mm
        young=jnp.ones(n_bonds),  # MPa
        area=jnp.ones(n_bonds)*(4*0.125**2),  # mm^2
        thermal_strain=-0.25*jnp.ones(n_bonds),
        damping=0.03*jnp.ones((n_points, 2)),
        masses=1.*jnp.ones((n_points, 2)),
        # NOTE: This is the initial guess for the material distribution
        # jnp.linspace(0, 1, n_bonds),  # NOTE: 0 means HTNI, 1 means LTNI
        phase=0.5*jnp.ones(n_bonds),
    )
    lattice = Lattice(
        connectivity=connectivity,
        control_params=control_params
    )
    problem = ForwardProblem(
        lattice=lattice,
        simulation_time=simulation_time,
        n_timepoints=sampled_temperatures.shape[0],
    )
    problem.setup(
        young_1_fn=young_1_fn,
        young_2_fn=young_2_fn,
        thermal_strain_1_fn=thermal_strain_1_fn,
        thermal_strain_2_fn=thermal_strain_2_fn,
        temperature_fn=temperature_fn,
    )

    # ### Target points

    # In[5]:

    # Select nodes on the boundary
    bottom_edge_ids = jnp.arange(0, n1)[::2]
    right_edge_ids = jnp.arange(n1, n_points-1, 2*n1+1)
    top_edge_ids = jnp.arange(n_points-n1, n_points, 1)[::-1][::2]
    left_edge_ids = jnp.arange(2*n1+1, n_points, 2*n1+1)[::-1]
    target_points_ids = jnp.concatenate([
        bottom_edge_ids,
        right_edge_ids,
        top_edge_ids,
        left_edge_ids,
    ])
    # lattice.control_params.reference_points[jnp.roll(target_points_ids, -n1-n2//4-1)]

    optimization_name = "circle_4_pointed_star"
    # Generate points on a circle starting from reference points
    center = lattice.control_params.reference_points.mean(axis=0)
    thetas = jnp.arctan2(lattice.control_params.reference_points[target_points_ids, 1] -
                         center[1], lattice.control_params.reference_points[target_points_ids, 0]-center[0])
    # thetas = jnp.linspace(thetas[0], thetas[0]+2*jnp.pi, thetas.shape[0])
    target1_points = jnp.array([
        center[0] + n1*spacing/2*jnp.cos(thetas),
        center[1] + n1*spacing/2*jnp.sin(thetas)
    ]).T
    # Generate a 4-pointed star
    diamond_spikeness = 0.1
    x_size = lattice.control_params.reference_points.max(
        axis=0)[0] - lattice.control_params.reference_points.min(axis=0)[0]
    y_size = lattice.control_params.reference_points.max(
        axis=0)[1] - lattice.control_params.reference_points.min(axis=0)[1]
    bottom_points = lattice.control_params.reference_points[bottom_edge_ids] + jnp.concatenate([
        jnp.linspace(
            0, 1, bottom_edge_ids.shape[0]//2+1)[:, None]*jnp.array([0, 1]),
        jnp.linspace(
            1, 0, bottom_edge_ids.shape[0]//2)[:, None]*jnp.array([0, 1]),
    ], axis=0)*x_size*diamond_spikeness
    right_points = lattice.control_params.reference_points[right_edge_ids] + jnp.concatenate([
        jnp.linspace(
            0, 1, right_edge_ids.shape[0]//2+1)[:, None]*jnp.array([-1, 0]),
        jnp.linspace(
            1, 0, right_edge_ids.shape[0]//2)[:, None]*jnp.array([-1, 0]),
    ], axis=0)*y_size*diamond_spikeness
    top_points = lattice.control_params.reference_points[top_edge_ids] + jnp.concatenate([
        jnp.linspace(
            0, 1, top_edge_ids.shape[0]//2+1)[:, None]*jnp.array([0, -1]),
        jnp.linspace(
            1, 0, top_edge_ids.shape[0]//2)[:, None]*jnp.array([0, -1]),
    ], axis=0)*x_size*diamond_spikeness
    left_points = lattice.control_params.reference_points[left_edge_ids] + jnp.concatenate([
        jnp.linspace(
            0, 1, left_edge_ids.shape[0]//2+1)[:, None]*jnp.array([1, 0]),
        jnp.linspace(
            1, 0, left_edge_ids.shape[0]//2)[:, None]*jnp.array([1, 0]),
    ], axis=0)*y_size*diamond_spikeness
    target2_points = jnp.concatenate([
        bottom_points,
        right_points,
        top_points,
        left_points,
    ], axis=0)

    # ### Run optimization

    # In[8]:

    optimization = OptimizationProblem(
        forward_problem=problem,
        target1_points=target1_points,
        target2_points=target2_points,
        target_points_ids=target_points_ids,
        target1_temperature=70.,
        target2_temperature=130.,
        weights=weights
    )
    optimization.run_optimization_nlopt(
        n_iterations=50,
        lower_bound=0.,
        upper_bound=1.,
    )
    # Save optimization data
    save_data(
        f"data/inverse_design/{optimization_name}/pareto/weights_{weights[0]:.2f}_{weights[1]:.2f}.pkl",
        optimization.to_data(),
    )


if __name__ == "__main__":

    # Run optimization sweep in parallel
    n_weights = 11
    weights_sweep = jnp.array([
        jnp.linspace(1, 0, n_weights),
        jnp.linspace(0, 1, n_weights),
    ]).T
    multiprocessing.set_start_method('spawn')
    with multiprocessing.Pool(processes=n_weights) as pool:
        pool.map(run_optimization, weights_sweep)
