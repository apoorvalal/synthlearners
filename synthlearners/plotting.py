import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Union, Tuple
from .synth import Synth, SynthResults


class SynthPlotter:
    """Plotting utilities for synthetic control results."""

    def __init__(
        self,
        figsize: Tuple[int, int] = (10, 6),
        style: Optional[str] = None,
        palette: Optional[dict] = None,
    ):
        """Initialize plotter with style settings.

        Args:
            figsize: Figure size (width, height)
            style: Optional matplotlib style
            palette: Optional color palette for different methods
        """
        self.figsize = figsize
        self.style = style
        self.default_palette = {
            "treated": "#000000",  # Black
            "did": "#66c2a5",  # Seafoam green (Set2)
            "controls": "grey",  #
            "lp constrained (p=1)": "#8da0cb",  # Blue-purple (Set2)
            "lp constrained (p=2)": "#e78ac3",  # Pink (Set2)
            "linear": "#5a8f2f",  # Darker green
            "simplex": "#b8a030",  # Darker yellow/gold
            "matching": "#a67d53",  # Darker tan/brown
            "treatment_line": "#e41a1c",  # Bright red
        }

        self.palette = palette or self.default_palette

    def plot_trajectories(
        self,
        results: Union[SynthResults, List[SynthResults]],
        Y: np.ndarray,
        treated_units: np.ndarray,
        T_pre: int,
        ax: Optional[plt.Axes] = None,
        show_controls: bool = True,
        title: Optional[str] = None,
        legend_title: str = "Imputed Counterfactuals",
        ylim: Optional[Tuple[float, float]] = None,
    ) -> plt.Axes:
        """Plot synthetic control trajectories.

        Args:
            results: Single or list of SynthResults objects
            Y: Original panel data array
            treated_units: Indices of treated units
            T_pre: Pre-treatment period cutoff
            ax: Optional matplotlib axes
            show_controls: Whether to show individual control unit trajectories
            title: Optional plot title
            legend_title: Title for the legend
            ylim: Optional y-axis limits

        Returns:
            matplotlib Axes object
        """
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=self.figsize)

        if self.style:
            plt.style.use(self.style)

        # Convert single result to list
        if isinstance(results, SynthResults):
            results = [results]

        # Get control units
        ctrl_units = np.setdiff1d(range(Y.shape[0]), treated_units)
        Y_control = Y[ctrl_units]

        # Plot treated unit
        ax.plot(
            results[0].treated_outcome,
            label="Treated",
            color=self.palette["treated"],
            linewidth=3,
        )

        # Plot naive DiD
        ax.plot(
            Y_control.mean(axis=0),
            label="(Naive) Control - DiD",
            color=self.palette["did"],
        )

        # Plot individual control trajectories
        if show_controls:
            ax.plot(
                Y_control.T,
                alpha=0.2,
                color=self.palette["controls"],
                linestyle="--",
                label="_nolegend_",
            )

        # Plot synthetic controls
        for result in results:
            method = result.method.value
            if method == "matrix_completion":
                label = "matrix_completion"
                if result.unit_intercept and result.time_intercept:
                    label += " w/Unit and Time FEs"
                if result.unit_intercept and not result.time_intercept:
                    label += " w/Unit FEs"
                if not result.unit_intercept and result.time_intercept:
                    label += " w/Time FEs"
            else:
                label = (
                    method if method != "lp_norm" else f"lp constrained (p={result.p})"
                )

            ax.plot(
                result.synthetic_outcome,
                alpha=0.6,
                label=label,
                color=self.palette.get(label, None),
            )

        # Add treatment line
        ax.axvline(
            T_pre,
            color=self.palette["treatment_line"],
            linestyle="--",
        )

        # Customize plot
        if title:
            ax.set_title(title)
        ax.legend(title=legend_title)
        if ylim:
            ax.set_ylim(ylim)

        return ax

    def plot_weights(
        self,
        results: Union[SynthResults, List[SynthResults]],
        ax: Optional[plt.Axes] = None,
        plot_type: str = "comparison",
        title: Optional[str] = None,
    ) -> plt.Axes:
        """Plot synthetic control weights.

        Args:
            results: Single or list of SynthResults objects
            ax: Optional matplotlib axes
            plot_type: Type of plot ('comparison' or 'distribution')
            title: Optional plot title

        Returns:
            matplotlib Axes object
        """
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=self.figsize)

        if isinstance(results, SynthResults):
            results = [results]

        if plot_type == "comparison":
            # Plot weights comparison
            for result in results:
                method = result.method.value
                if method == "matrix_completion":
                    label = "matrix_completion"
                    if result.unit_intercept and result.time_intercept:
                        label += " w/Unit and Time FEs"
                    if result.unit_intercept and not result.time_intercept:
                        label += " w/Unit FEs"
                    if not result.unit_intercept and result.time_intercept:
                        label += " w/Time FEs"
                else:
                    label = (
                        method if method != "lp_norm" else f"lp constrained (p={result.p})"
                    )
                ax.plot(
                    result.unit_weights,
                    label=label,
                    marker=".",
                    alpha=0.6,
                    color=self.palette.get(label, None),
                )

            ax.axhline(0, color="black", linestyle=":", alpha=0.5)
            ax.set_title(title or "Comparison of Synthetic Control Weights")
            ax.legend()

        elif plot_type == "distribution":
            # Plot weight distributions
            for result in results:
                method = result.method.value
                if method == "matrix_completion":
                    label = "matrix_completion"
                    if result.unit_intercept and result.time_intercept:
                        label += " w/Unit and Time FEs"
                    if result.unit_intercept and not result.time_intercept:
                        label += " w/Unit FEs"
                    if not result.unit_intercept and result.time_intercept:
                        label += " w/Time FEs"
                else:
                    label = (
                        method if method != "lp_norm" else f"lp constrained (p={result.p})"
                    )
                ax.hist(
                    result.unit_weights,
                    alpha=0.6,
                    label=label,
                    color=self.palette.get(label, None),
                    bins=30,
                )

            ax.set_title(title or "Distribution of Synthetic Control Weights")
            ax.legend()

        return ax


def plot_simulation_results(
    simulator_results: List[Tuple],
    synth_results: List[List[SynthResults]],
    T_pre: int,
    figsize: Tuple[int, float] = (12, 8),
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot results from multiple simulations.

    Args:
        simulator_results: List of (Y, L, treated_units) tuples
        synth_results: List of lists of SynthResults objects
        T_pre: Pre-treatment period cutoff
        figsize: Figure size

    Returns:
        Figure and Axes objects
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    # Plot average treatment effects
    plotter = SynthPlotter()

    # Average across simulations
    avg_effects = {}
    for method in synth_results[0][0].method.__class__:
        effects = []
        for sim_results in synth_results:
            for result in sim_results:
                if result.method == method:
                    effects.append(result.treatment_effect())
        avg_effects[method] = np.mean(effects, axis=0)

    # Plot average effects
    for method, effect in avg_effects.items():
        ax1.plot(effect, label=method.value, alpha=0.6)

    ax1.axvline(T_pre, color="red", linestyle="--", label="Treatment")
    ax1.set_title("Average Treatment Effects Across Simulations")
    ax1.legend()

    # Plot distribution of pre-RMSE
    for method in synth_results[0][0].method.__class__:
        rmse_values = []
        for sim_results in synth_results:
            for result in sim_results:
                if result.method == method:
                    rmse_values.append(result.pre_treatment_rmse)

        ax2.hist(rmse_values, alpha=0.6, label=method.value, bins=30)

    ax2.set_title("Distribution of Pre-Treatment RMSE")
    ax2.legend()

    fig.tight_layout()
    return fig, (ax1, ax2)
