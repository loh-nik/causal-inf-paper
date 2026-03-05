"""
plot_graph_multilags.py
=======================
Drop-in extension for tigramite's plotting.py that draws each active time lag
between a node pair as its own curved arrow, optionally with a distinct color
per lag.

Usage
-----
    from plot_graph_multilags import plot_graph_multilags

    plot_graph_multilags(
        graph=graph,           # shape (N, N, tau_max+1), string dtype
        val_matrix=val_matrix, # shape (N, N, tau_max+1), float
        lag_colors={1: "tab:blue", 2: "tab:red"},  # or a list, or None
        curved_radius_base=0.2,
        curved_radius_step=0.2,
        var_names=["X0", "X1", "X2"],
        **plot_graph_kwargs,   # any kwarg accepted by tigramite's plot_graph
    )

How it works
------------
The original plot_graph collapses all lagged edges for a given (u, v) pair
into one arrow by selecting the lag with the largest absolute val_matrix entry.
This wrapper calls plot_graph once for the contemporaneous (lag-0) edges, then
once per active lag >= 1, each time passing a graph that contains only that
lag's edges. Each call uses a progressively larger curved_radius so the arrows
fan out visibly. When lag_colors is provided, newly added FancyArrowPatch
artists are identified via a before/after snapshot and repainted to the
requested color.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

try:
    from tigramite.plotting import plot_graph
except ImportError:
    from plotting import plot_graph


def plot_graph_multilags(
    graph,
    val_matrix=None,
    lag_colors=None,
    curved_radius_base=0.2,
    curved_radius_step=0.15,
    arrow_linewidth=8.0,
    fig_ax=None,
    figsize=None,
    var_names=None,
    save_name=None,
    show_colorbar=True,
    show_legend=True,
    legend_fontsize=9,
    **plot_graph_kwargs,
):
    """Draw a tigramite graph where every active lag gets its own arrow.

    Parameters
    ----------
    graph : np.ndarray, shape (N, N, tau_max+1), dtype '<U3'
        Tigramite graph array. Lag-0 entries are plotted once as straight
        (contemporaneous) edges. Each lag tau >= 1 that has at least one
        non-empty entry is plotted as a separate curved arrow.
    val_matrix : np.ndarray, shape (N, N, tau_max+1), optional
        Strength values. Used for arrow color when no per-lag color is given.
    lag_colors : dict, list, or None
        Per-lag matplotlib colors.
        - dict:  {lag_index: color}, e.g. {1: 'tab:blue', 2: 'tab:red'}
        - list:  colors in lag order starting from lag 1, e.g. ['tab:blue', 'tab:red']
        - None:  use the shared colormap from val_matrix (each arrow is
                 colored by its own val_matrix value).
    curved_radius_base : float
        Curvature radius of the first lagged arrow (lag 1).
    curved_radius_step : float
        Extra curvature added for each subsequent lag so arrows fan out.
    arrow_linewidth : float
        Linewidth passed to plot_graph for every call.
    fig_ax : tuple (fig, ax) or None
        Existing figure and axes to draw on. If None, a new figure is created.
    figsize : tuple or None
        Figure size, used only when fig_ax is None.
    var_names : list or None
        Variable names passed to plot_graph.
    save_name : str or None
        If given, the figure is saved to this path after all lags are drawn.
    show_colorbar : bool
        Whether to show a colorbar. Only shown on the final lag call to avoid
        duplicates. Has no effect when lag_colors is provided.
    show_legend : bool
        If True and lag_colors is supplied, a legend with lag labels is added.
    legend_fontsize : int
        Font size of the legend.
    **plot_graph_kwargs
        Additional keyword arguments forwarded to every plot_graph call.
        Common ones: vmin_edges, vmax_edges, edge_ticks, cmap_edges,
        node_size, arrowhead_size, node_pos, etc.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    graph = np.copy(graph.squeeze())

    if graph.ndim == 4:
        raise ValueError(
            "4-D TSG arrays are not supported. Use plot_time_series_graph instead."
        )

    if graph.ndim == 2:
        # Non-temporal graph — delegate directly to plot_graph.
        return plot_graph(
            graph=graph,
            val_matrix=val_matrix,
            fig_ax=fig_ax,
            figsize=figsize,
            var_names=var_names,
            save_name=save_name,
            show_colorbar=show_colorbar,
            arrow_linewidth=arrow_linewidth,
            curved_radius=curved_radius_base,
            **plot_graph_kwargs,
        )

    _, _, tau_max_plus1 = graph.shape
    tau_max = tau_max_plus1 - 1

    # --- Resolve per-lag colors ---
    _lag_color_map = {}
    if lag_colors is not None:
        if isinstance(lag_colors, dict):
            _lag_color_map = {int(k): v for k, v in lag_colors.items()}
        elif isinstance(lag_colors, (list, tuple)):
            _lag_color_map = {idx + 1: c for idx, c in enumerate(lag_colors)}
        else:
            raise ValueError("lag_colors must be a dict, list, or None.")

    active_lags = [
        tau for tau in range(1, tau_max + 1)
        if np.any(graph[:, :, tau] != 0)
    ]

    # --- Set up figure/axes ---
    if fig_ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, frame_on=False)
        fig_ax_current = (fig, ax)
    else:
        fig, ax = fig_ax
        fig_ax_current = fig_ax

    # Pop kwargs that we manage explicitly so they aren't passed twice.
    vmin_edges = plot_graph_kwargs.pop("vmin_edges", -1.0)
    vmax_edges = plot_graph_kwargs.pop("vmax_edges", 1.0)
    cmap_edges = plot_graph_kwargs.pop("cmap_edges", "RdBu_r")
    show_auto_colorbar = plot_graph_kwargs.pop("show_auto_colorbar", True)

    def _graph_for_lag(tau):
        """Graph slice with only lag-0 and a single lagged tau."""
        g = np.full_like(graph, 0)
        g[:, :, 0] = graph[:, :, 0]
        for i in range(g.shape[0]):
            g[i,i,:] = graph[i,i,:]
        g[:, :, tau] = graph[:, :, tau]
        return g

    def _val_for_lag(tau):
        """val_matrix slice with only lag-0 and a single lagged tau."""
        if val_matrix is None:
            return None
        v = np.zeros_like(val_matrix)
        v[:, :, 0] = val_matrix[:, :, 0]
        v[:, :, tau] = val_matrix[:, :, tau]
        return v

    def _arrow_snapshot():
        """IDs of all current FancyArrowPatch artists on ax."""
        return {id(a) for a in ax.get_children() if isinstance(a, FancyArrowPatch)}

    # --- Pass 1: contemporaneous (lag-0) edges and nodes ---
    g_contemp = np.full_like(graph, 0)
    g_contemp[:, :, 0] = graph[:, :, 0]
    v_contemp = None
    if val_matrix is not None:
        v_contemp = np.zeros_like(val_matrix)
        v_contemp[:, :, 0] = val_matrix[:, :, 0]

    plot_graph(
        graph=g_contemp,
        # val_matrix=v_contemp,
        val_matrix = val_matrix,
        fig_ax=fig_ax_current,
        var_names=var_names,
        arrow_linewidth=arrow_linewidth,
        curved_radius=curved_radius_base,
        vmin_edges=vmin_edges,
        vmax_edges=vmax_edges,
        cmap_edges=cmap_edges if not _lag_color_map else None,
        show_colorbar=False,
        show_auto_colorbar=show_auto_colorbar,
        **plot_graph_kwargs,
    )

    # --- Pass 2: one plot_graph call per active lag ---
    for lag_idx, tau in enumerate(active_lags):
        color_override = _lag_color_map.get(tau)
        radius = curved_radius_base + lag_idx * curved_radius_step
        is_last = tau == active_lags[-1]

        before = _arrow_snapshot()

        plot_graph(
            graph=_graph_for_lag(tau),
            # val_matrix=None if color_override else _val_for_lag(tau),
            val_matrix = val_matrix,
            fig_ax=fig_ax_current,
            var_names=var_names,
            arrow_linewidth=arrow_linewidth,
            curved_radius=radius,
            cmap_edges=None if color_override else cmap_edges,
            # cmap_nodes=None,
            vmin_edges=vmin_edges,
            vmax_edges=vmax_edges,
            show_colorbar=show_colorbar and is_last and color_override is None,
            show_auto_colorbar=False,
            **plot_graph_kwargs,
        )

        if color_override is not None:
            new_ids = _arrow_snapshot() - before
            for artist in ax.get_children():
                if id(artist) not in new_ids:
                    continue
                alpha = artist.get_alpha()
                if alpha is not None and alpha < 1e-6:
                    continue  # skip invisible marker patches
                artist.set_facecolor(color_override)
                artist.set_edgecolor(color_override)

    # --- Legend ---
    if show_legend and _lag_color_map:
        handles = [
            mpatches.Patch(color=_lag_color_map[tau], label=f"lag {tau}")
            for tau in sorted(_lag_color_map)
        ]
        ax.legend(handles=handles, fontsize=legend_fontsize,
                  loc="upper right", framealpha=0.8)

    if save_name is not None:
        plt.savefig(save_name, dpi=300, bbox_inches="tight")

    return fig, ax


if __name__ == "__main__":
    N, tau_max = 3, 2
    graph = np.zeros((N, N, tau_max + 1), dtype="<U3")
    graph[:] = ""

    # Node 0 -> Node 1 at lag 1 AND lag 2 — the key multi-lag scenario
    graph[0, 1, 1] = "-->"
    graph[0, 1, 2] = "-->"
    # Node 1 -> Node 2 at lag 1 only
    graph[1, 2, 1] = "-->"
    # Contemporaneous: Node 0 -> Node 2
    graph[0, 2, 0] = "-->"
    graph[2, 0, 0] = "<--"

    val_matrix = np.zeros((N, N, tau_max + 1))
    val_matrix[0, 1, 1] = 0.6
    val_matrix[0, 1, 2] = -0.4
    val_matrix[1, 2, 1] = 0.3
    val_matrix[0, 2, 0] = val_matrix[2, 0, 0] = 0.5

    fig, ax = plot_graph_multilags(
        graph=graph,
        val_matrix=val_matrix,
        lag_colors={1: "tab:blue", 2: "tab:red"},
        curved_radius_base=0.2,
        curved_radius_step=0.2,
        var_names=["X0", "X1", "X2"],
        show_colorbar=False,
        show_legend=True,
        figsize=(6, 5),
    )

    plt.tight_layout()
    plt.savefig("multilags_demo.pdf", dpi=150)
    print("Saved multilags_demo.pdf")
