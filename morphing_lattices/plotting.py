from pathlib import Path
import matplotlib.animation as animation
from matplotlib import cm
import matplotlib
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from morphing_lattices.structure import Lattice
from morphing_lattices.kinematics import rotation_matrix
import jax.numpy as jnp


def plot_lattice(lattice: Lattice, displacement=None, xlim=None, ylim=None, title="Lattice", x_label=None, y_label=None, figsize=(5, 5), annotate=False, bond_values=None, bond_color=None, legend_labels=None, legend_colors=None, LTNI_bond_indices=None, HTNI_bond_indices=None, node_size=None, legend_label=None, fontsize=14, cmap="coolwarm", axis=True, grid=True, lattice_line_width=2,):
    points = lattice.control_params.reference_points if displacement is None else displacement + \
        lattice.control_params.reference_points
    connectivity = lattice.connectivity

    # Plot the lattice
    fig, ax = plt.subplots(constrained_layout=True, figsize=figsize)
    ax.set_aspect("equal")
    if x_label is not None:
        ax.set_xlabel(x_label, fontsize=fontsize)
    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    ax.tick_params(labelsize=0.8*fontsize)
    if not axis:
        ax.axis("off")
    if not grid:
        ax.grid(False)

    # Plot nodes
    ax.scatter(points[:, 0], points[:, 1],
               color="black", zorder=10, s=node_size)
    # Plot the bonds
    if bond_values is not None:
        # Color the bonds
        _cmap = matplotlib.colormaps[cmap]
        norm = plt.Normalize(vmin=bond_values.min(), vmax=bond_values.max())
        colors = _cmap(norm(bond_values))
        # Add a colorbar
        sm = plt.cm.ScalarMappable(cmap=_cmap, norm=norm)
        sm.set_array(bond_values)
        cb = fig.colorbar(sm, ax=ax, pad=0.04, aspect=25)
        cb.ax.tick_params(labelsize=0.8*fontsize)
        cb.ax.set_ylabel(
            legend_label if legend_label is not None else "", fontsize=fontsize)
    else:
        colors = bond_color if bond_color is not None else "#2980b9"

    cntr_HTNI = 0
    cntr_LTNI = 0
    if LTNI_bond_indices is not None and HTNI_bond_indices is not None:
        for i, pair in enumerate(connectivity):
            if i in LTNI_bond_indices:
                if cntr_LTNI == 0:
                    # ax.plot(
                    #     *points[pair].T, lw=2, color="#069AF3" if colors is None else colors[i], label='LTNI')
                    ax.plot(
                        *points[pair].T, lw=2, color="#069AF3", label='LTNI')
                    cntr_LTNI = cntr_LTNI+1
                else:
                    # ax.plot(
                    #     *points[pair].T, lw=2, color="#069AF3" if colors is None else colors[i])
                    ax.plot(
                        *points[pair].T, lw=2, color="#069AF3")
            elif i in HTNI_bond_indices:
                if cntr_HTNI == 0:
                    # ax.plot(
                    #     *points[pair].T, lw=2, color="#F97306" if colors is None else colors[i], label='HTNI')
                    ax.plot(
                        *points[pair].T, lw=2, color="#F97306", label='HTNI')
                    cntr_HTNI = cntr_HTNI+1
                else:
                    # ax.plot(
                    #     *points[pair].T, lw=2, color="#F97306" if colors is None else colors[i])
                    ax.plot(
                        *points[pair].T, lw=2, color="#F97306")

        ax.legend()
    else:
        collection_bonds = LineCollection(
            points[connectivity], color=colors, linewidth=lattice_line_width)
        ax.add_collection(collection_bonds)
        # Add legend
        if legend_labels is not None and legend_colors is not None:
            custom_lines = [Line2D([0], [0], color=legend_colors[i], lw=lattice_line_width)
                            for i in range(len(legend_labels))]
            ax.legend(custom_lines, legend_labels,
                      fontsize=fontsize, loc="upper right")

    if annotate:
        for i, pair in enumerate(connectivity):
            ax.annotate(f"{i}", points[pair].mean(axis=0), color="r")
        for id, point in enumerate(points):
            ax.annotate(f"{id}", point)

    # xylim
    _xlim, _ylim = lattice.get_xy_limits()
    ax.set(xlim=_xlim if xlim is None else xlim,
           ylim=_ylim if ylim is None else ylim)

    return fig, ax


def generate_animation(lattice: Lattice, solution: jnp.ndarray, lattice_number=None, out_filename=None, rotated_points=False, frame_range=None, figsize=None, xlim=None, ylim=None, fps=20, dpi=200, title=None, x_label=None, y_label=None, legend_label=None, bond_values=None, bond_color=None, legend_labels=None, legend_colors=None, node_size=None, fontsize=14, cmap="coolwarm", axis=True, grid=True, lattice_line_width=2, target_points=None, target_color="black", target_node_size=5, target_line_width=2):

    tick_size = 0.8*fontsize
    # Plot the lattice
    fig, ax = plt.subplots(constrained_layout=True, figsize=figsize)
    ax.set_aspect("equal")
    if x_label is not None:
        ax.set_xlabel(x_label, fontsize=fontsize)
    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=fontsize)
    if title is not None:
        ax.set_title(title, fontsize=fontsize)
    ax.tick_params(labelsize=tick_size)
    if not axis:
        ax.axis("off")
    if not grid:
        ax.grid(False)

    if lattice_number is not None:
        ax.add_patch(matplotlib.patches.Rectangle((xlim[0], ylim[0]+0.9*(ylim[1] - ylim[0])), 0.0*(
            xlim[1] - xlim[0]), 0.0*(ylim[1] - ylim[0]), color='#F97306', label='HTNI'))
        ax.add_patch(matplotlib.patches.Rectangle((xlim[0], ylim[0]+0.8*(ylim[1] - ylim[0])), 0.0*(
            xlim[1] - xlim[0]), 0.0*(ylim[1] - ylim[0]), color='#069AF3', label='LTNI'))
        ax.legend()

    # First frame
    connectivity = lattice.connectivity
    if rotated_points and lattice_number == 10:
        points = jnp.dot(rotation_matrix(-jnp.pi/2),
                         (lattice.control_params.reference_points + solution[0, 0]).T).T
        points0 = points.copy()
        points = points.at[:, 1].set(points[:, 1]-points0[9, 1])
        # Nodes
        scatter_plot = ax.scatter(
            points[:, 0], points[:, 1], color="black", zorder=10, s=node_size)
    else:
        points = lattice.control_params.reference_points + solution[0, 0]
        # Nodes
        scatter_plot = ax.scatter(
            points[:, 0], points[:, 1], color="black", zorder=10, s=node_size)

    # Bonds
    collection_bonds = LineCollection(
        points[connectivity], color="black", linewidth=lattice_line_width)
    ax.add_collection(collection_bonds)
    if bond_color is not None:
        # Color the bonds with a single color
        collection_bonds.set_color(bond_color)
    elif bond_values is not None:
        # Color the bonds according to the bond values
        _cmap = matplotlib.colormaps[cmap]
        norm = plt.Normalize(vmin=bond_values.min(), vmax=bond_values.max())
        colors = _cmap(norm(bond_values[0]))
        collection_bonds.set_color(colors)
        # Add colorbar
        cb = fig.colorbar(
            cm.ScalarMappable(cmap=cmap, norm=norm),
            ax=ax,
            pad=0.04,
            aspect=25
        )
        cb.ax.tick_params(labelsize=tick_size)
        cb.ax.set_ylabel(legend_label, fontsize=fontsize)
    # xylim
    _xlim, _ylim = lattice.get_xy_limits()
    ax.set(xlim=_xlim if xlim is None else xlim,
           ylim=_ylim if ylim is None else ylim)

    # Add legend
    if legend_labels is not None and legend_colors is not None:
        custom_lines = [Line2D([0], [0], color=legend_colors[i], lw=lattice_line_width)
                        for i in range(len(legend_labels))]
        ax.legend(custom_lines, legend_labels,
                  fontsize=fontsize, loc="upper right")

    # Add target points
    if target_points is not None:
        if type(target_points) == list or type(target_points) == tuple:
            target_color = [target_color]*len(target_points) if type(
                target_color) != list and type(target_color) != tuple else target_color
            for points, color in zip(target_points, target_color):
                points = jnp.concatenate([points, points[0, None]], axis=0)
                ax.plot(points[:, 0], points[:, 1], "-o",
                        color=color, zorder=20, markersize=target_node_size, lw=target_line_width)
        else:
            target_points = jnp.concatenate(
                [target_points, target_points[0, None]], axis=0)
            ax.plot(target_points[:, 0], target_points[:, 1], "-o",
                    color=target_color, zorder=20, markersize=target_node_size, lw=target_line_width)

    def animate(i):
        # Update nodes
        if rotated_points and lattice_number == 10:
            points = jnp.dot(rotation_matrix(-jnp.pi/2),
                             (lattice.control_params.reference_points + solution[i, 0]).T).T
            points = points.at[:, 1].set(points[:, 1] - points0[9, 1])
            scatter_plot.set_offsets(points)
        else:
            points = lattice.control_params.reference_points + solution[i, 0]
            scatter_plot.set_offsets(points)
        # Update bonds
        collection_bonds.set_segments(points[connectivity])
        # Update bond values
        if bond_values is not None and bond_color is None:
            # Color the bonds
            colors = _cmap(norm(bond_values[i]))
            collection_bonds.set_color(colors)
        return scatter_plot, collection_bonds

    frames = range(solution.shape[0]) if frame_range is None else frame_range
    anim = animation.FuncAnimation(
        fig, animate, frames=frames, blit=True)  # type: ignore
    out_path = Path(f"{out_filename}.mp4")
    # Make sure parents directories exist
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out_path), writer="ffmpeg", fps=fps, dpi=dpi)
