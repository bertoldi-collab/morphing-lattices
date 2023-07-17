import matplotlib.pyplot as plt
from morphing_lattices.structure import Lattice


def plot_lattice(lattice: Lattice, displacement=None, xlim=None, ylim=None, title="Lattice", figsize=(5, 5), annotate=False, bond_values=None, label=None, fontsize=14, cmap="coolwarm", axis=True):
    points = lattice.control_params.reference_points if displacement is None else displacement + lattice.control_params.reference_points
    connectivity = lattice.connectivity

    # Plot the lattice
    fig, ax = plt.subplots(constrained_layout=True, figsize=figsize)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$", fontsize=fontsize)
    ax.set_ylabel("$y$", fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    ax.tick_params(labelsize=0.8*fontsize)
    if not axis:
        ax.axis("off")

    # Plot nodes
    ax.scatter(points[:, 0], points[:, 1], color="black", zorder=10)
    # Plot the bonds
    if bond_values is not None:
        # Color the bonds
        _cmap = plt.cm.get_cmap(cmap)
        norm = plt.Normalize(vmin=bond_values.min(), vmax=bond_values.max())
        colors = _cmap(norm(bond_values))
        # Add a colorbar
        sm = plt.cm.ScalarMappable(cmap=_cmap, norm=norm)
        sm.set_array(bond_values)
        cb = fig.colorbar(sm, ax=ax, pad=0.04, aspect=25)
        cb.ax.tick_params(labelsize=0.8*fontsize)
        cb.ax.set_ylabel(label if label is not None else "", fontsize=fontsize)
    else:
        colors = None
    for i, pair in enumerate(connectivity):
        ax.plot(*points[pair].T, lw=2, color="#2980b9" if colors is None else colors[i])

    if annotate:
        for i, pair in enumerate(connectivity):
            ax.annotate(f"{i}", points[pair].mean(axis=0), color="r")
        for id, point in enumerate(points):
            ax.annotate(f"{id}", point)

    # xylim
    x_ext = points[:, 0].max() - points[:, 0].min()
    y_ext = points[:, 1].max() - points[:, 1].min()
    if xlim is None:
        ax.set_xlim(points[:, 0].min() - 0.05*x_ext, points[:, 0].max() + 0.05*x_ext)
    else:
        ax.set_xlim(*xlim)
    if ylim is None:
        ax.set_ylim(points[:, 1].min() - 0.05*y_ext, points[:, 1].max() + 0.05*y_ext)
    else:
        ax.set_ylim(*ylim)

    return fig, ax
