# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec


# %%
def save_plot(figure, filename):
    """
    Save a matplotlib figure to a file.

    Parameters:
        figure (matplotlib.figure.Figure): The figure to save.
        filename (str): The name of the file to save the figure as.
    """
    figure.savefig(
        filename, dpi=350, format="png", bbox_inches="tight", transparent=True
    )


# %%
def heatmap(codf, datadf, filename=None):
    plot_df = codf.merge(
        datadf[["material_id", "formation_energy_per_atom", "energy_above_hull"]],
        on="material_id",
    )
    x = plot_df.predScore
    y = plot_df.energy_above_hull
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    xzoomed = plot_df.predScore[plot_df.energy_above_hull <= 1]
    yzoomed = plot_df.energy_above_hull[plot_df.energy_above_hull <= 1]
    xzoomed = xzoomed[~np.isnan(xzoomed)]
    yzoomed = yzoomed[~np.isnan(yzoomed)]

    # Calculate the bins for the full dataset
    hist_full, xedges, yedges = np.histogram2d(x, y, bins=(50, 50))

    # Use the full data set to find the common vmin and vmax
    vmin = 1  # Since you use cmin=1, we start the colorbar at 1
    vmax = hist_full.max()  # The max count from the full data set

    # First plot
    plt.hist2d(
        x, y, bins=(50, 50), cmap="viridis", norm=LogNorm(vmin=vmin, vmax=vmax), cmin=1
    )
    cbar = plt.colorbar()
    plt.xlabel("predScore")
    plt.ylabel("energy_above_hull")
    plt.title("Density Heatmap")
    if filename:
        save_plot(plt.gcf(), filename)
    plt.show()


# %%
def heatmapZoom(codf, datadf, ehull_cutoff=1, filename=None):
    plot_df = codf.merge(
        datadf[["material_id", "formation_energy_per_atom", "energy_above_hull"]],
        on="material_id",
    )
    x = plot_df.predScore
    y = plot_df.energy_above_hull
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    # Calculate the bins for the full dataset
    hist_full, _, _ = np.histogram2d(x, y, bins=(50, 50))
    # Use the full data set to find the common vmin and vmax
    vmin = 1  # Since you use cmin=1, we start the colorbar at 1
    vmax = hist_full.max()  # The max count from the full data set

    xzoomed = plot_df.predScore[plot_df.energy_above_hull <= ehull_cutoff]
    yzoomed = plot_df.energy_above_hull[plot_df.energy_above_hull <= ehull_cutoff]
    xzoomed = xzoomed[~np.isnan(xzoomed)]
    yzoomed = yzoomed[~np.isnan(yzoomed)]

    plt.hist2d(
        xzoomed,
        yzoomed,
        bins=(50, 50),
        cmap="viridis",
        norm=LogNorm(vmin=vmin, vmax=vmax),
        cmin=1,
    )
    cbar = plt.colorbar()
    plt.xlabel("predScore")
    plt.ylabel("energy_above_hull")
    plt.title("Zoomed in Density Heatmap")
    if filename:
        save_plot(plt.gcf(), filename)
    plt.show()


# %%
# Assuming tdf.predScore and tdf.energy_above_hull are your data columns
def density_colors(x, y):
    # Filter out NaN values
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    # Create a 2D histogram to compute color mapping
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=(50, 50))

    # Calculate color for each data point
    x_indices = np.clip(np.digitize(x, xedges) - 1, 0, heatmap.shape[0] - 1)
    y_indices = np.clip(np.digitize(y, yedges) - 1, 0, heatmap.shape[1] - 1)
    colors = heatmap[x_indices, y_indices]

    return colors, x, y


# %%
def density_colors_with_contours(x, y):
    # Filter out NaN values
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    # Create a 2D histogram to compute color mapping
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=(50, 50))

    # Calculate color for each data point
    x_indices = np.clip(np.digitize(x, xedges) - 1, 0, heatmap.shape[0] - 1)
    y_indices = np.clip(np.digitize(y, yedges) - 1, 0, heatmap.shape[1] - 1)
    colors = heatmap[x_indices, y_indices]

    return colors, x, y, heatmap, xedges, yedges


# %%
def scatter_hm(codf, datadf, filename=None):
    plot_df = codf.merge(
        datadf[["material_id", "formation_energy_per_atom", "energy_above_hull"]],
        on="material_id",
    )
    colors, x, y = density_colors(plot_df.predScore, plot_df.energy_above_hull)
    # Plot first scatter plot
    plt.scatter(x, y, c=colors, cmap="viridis", norm=LogNorm(), alpha=0.7, s=20)
    plt.colorbar()
    plt.xlabel("predScore")
    plt.ylabel("energy_above_hull")
    plt.title("Density Scatter Plot")
    if filename:
        save_plot(plt.gcf(), filename)
    plt.show()


# %%
def scatter_hm_zoomed(codf, datadf, ehull_cutoff=1, filename=None):
    plot_df = codf.merge(
        datadf[["material_id", "formation_energy_per_atom", "energy_above_hull"]],
        on="material_id",
    )
    x = plot_df.predScore
    y = plot_df.energy_above_hull
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    # Calculate the bins for the full dataset
    hist_full, _, _ = np.histogram2d(x, y, bins=(50, 50))
    # Use the full data set to find the common vmin and vmax
    vmin = 1  # Since you use cmin=1, we start the colorbar at 1
    vmax = hist_full.max()  # The max count from the full data set

    x_zoomed = plot_df.predScore[plot_df.energy_above_hull <= ehull_cutoff]
    y_zoomed = plot_df.energy_above_hull[plot_df.energy_above_hull <= ehull_cutoff]
    x_zoomed = x_zoomed[~np.isnan(x_zoomed)]
    y_zoomed = y_zoomed[~np.isnan(y_zoomed)]
    heatmap, xedges, yedges = np.histogram2d(x_zoomed, y_zoomed, bins=(50, 50))

    # We reuse the original bins and heatmap to ensure consistent color mapping
    x_indices_zoomed = np.clip(
        np.digitize(x_zoomed, xedges) - 1, 0, heatmap.shape[0] - 1
    )
    y_indices_zoomed = np.clip(
        np.digitize(y_zoomed, yedges) - 1, 0, heatmap.shape[1] - 1
    )
    colors_zoomed = heatmap[x_indices_zoomed, y_indices_zoomed]

    # Plot second (zoomed-in) scatter plot
    plt.scatter(
        x_zoomed,
        y_zoomed,
        c=colors_zoomed,
        cmap="viridis",
        norm=LogNorm(vmin=vmin, vmax=vmax),
        alpha=0.7,
        s=20,
    )
    plt.colorbar()
    plt.xlabel("predScore (Zoomed)")
    plt.ylabel("energy_above_hull (Zoomed)")
    plt.title("Zoomed Density Scatter Plot")
    if filename:
        save_plot(plt.gcf(), filename)
    plt.show()


# %%
def label_dist(codf, datadf, ehull=False, prop="synth", filename=None):
    plot_df = codf.merge(
        datadf[["material_id", "formation_energy_per_atom", "energy_above_hull"]],
        on="material_id",
    )
    if ehull:
        prop = "stability"
    prediction = "prediction"
    edf = plot_df[plot_df[prop] == 1]
    tdf = plot_df[plot_df[prop] == 0]
    edf = edf.sort_values(prediction, ascending=False)
    tdf = tdf.sort_values(prediction, ascending=True)

    print(
        f"edf prediction means is {edf.prediction.mean()}. Length of edf is {len(edf)}."
    )
    print(
        f"tdf prediction means is {tdf.prediction.mean()}. Length of tdf is {len(tdf)}."
    )

    data_classes_e = ["False (-) class", "True (+) class"]
    data_classes_t = ["Predicted (-) class", "Predicted (+) class"]

    colors = ListedColormap(["r", "b"])

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 7))

    scatter = ax1.scatter(
        edf.energy_above_hull,
        edf.formation_energy_per_atom,
        c=edf["prediction"],  # Make sure 'prediction' is quoted if it's a column name
        cmap=colors,
        alpha=0.6,
    )

    ax1.legend(handles=scatter.legend_elements()[0], labels=data_classes_e)
    ax1.set_title(
        "Prediction Distribution of Experimental Data", fontsize=13.5, fontweight="bold"
    )

    # Use ax2 directly to create the scatter plot instead of calling plt.subplot again
    scatter2 = ax2.scatter(
        tdf.energy_above_hull,
        tdf.formation_energy_per_atom,
        c=tdf["prediction"],  # Again, quote 'prediction' if it's a column name
        cmap=colors,
        alpha=0.5,
    )

    # If you want to modify tick parameters for ax1, do it with ax1.tick_params
    ax1.tick_params("x", labelbottom=False)

    # Set the title for ax2 as well, and any other configurations you need
    ax2.set_title(
        "Prediction Distribution of Theoretical Data", fontsize=13.5, fontweight="bold"
    )

    ax2.legend(handles=scatter.legend_elements()[0], labels=data_classes_t)

    ax2.set_xlabel("Energy above hull", fontsize=15)
    # ax2.set_ylabel('Formation energy per atom', fontsize=12)
    ax2.set_title(
        "Prediction Distribution of Theorertical Data", fontsize=13.5, fontweight="bold"
    )
    plt.subplots_adjust(hspace=0.15)
    fig.text(
        0.04,
        0.5,
        "Formation energy per atom",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=16.5,
    )
    if filename:
        save_plot(figure=fig, filename=filename)


# %%


def label_dist2(codf, datadf, prop="synth", ehull=False, filename=None):
    plot_df = codf.merge(
        datadf[["material_id", "formation_energy_per_atom", "energy_above_hull"]],
        on="material_id",
    )
    if ehull:
        prop = "stability"
    edf = plot_df[plot_df[prop] == 1]
    tdf = plot_df[plot_df[prop] == 0]
    edf = edf.sort_values("prediction", ascending=False)
    tdf = tdf.sort_values("prediction", ascending=True)

    colors = ListedColormap(["r", "b"])
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))  # 2x2 subplot grid

    # Define the plots for experimental data
    scatter1 = axs[0, 0].scatter(
        edf.energy_above_hull[edf["prediction"] == 0],
        edf.formation_energy_per_atom[edf["prediction"] == 0],
        c=edf["prediction"][edf["prediction"] == 0],
        cmap=colors,
        alpha=0.6,
    )

    axs[0, 0].set_title(
        "Experimental Data (Negative Label)", fontsize=13.5, fontweight="bold"
    )

    scatter2 = axs[1, 0].scatter(
        edf.energy_above_hull[
            edf["prediction"] == 1
        ],  # Filter for minority class if needed
        edf.formation_energy_per_atom[edf["prediction"] == 1],
        c="blue",
        alpha=0.6,
    )

    axs[1, 0].set_title(
        "Experimental Data (Positive Label)", fontsize=13.5, fontweight="bold"
    )

    # Define the plots for theoretical data
    scatter3 = axs[0, 1].scatter(
        tdf.energy_above_hull[tdf["prediction"] == 0],
        tdf.formation_energy_per_atom[tdf["prediction"] == 0],
        c=tdf["prediction"][tdf["prediction"] == 0],
        cmap=colors,
        alpha=0.5,
    )

    axs[0, 1].set_title(
        "Theoretical Data (Negative Label)", fontsize=13.5, fontweight="bold"
    )

    scatter4 = axs[1, 1].scatter(
        tdf.energy_above_hull[tdf["prediction"] == 1],
        tdf.formation_energy_per_atom[tdf["prediction"] == 1],
        c="b",
        alpha=0.5,
    )

    axs[1, 1].set_title(
        "Theoretical Data (Positive Label)", fontsize=13.5, fontweight="bold"
    )

    # Set common labels
    for ax in axs.flat:
        # ax.set(xlabel='Energy above hull', ylabel='Formation energy per atom')
        ax.label_outer()  # Only show outer labels

    # Adjust subplot params
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    # Add a big axis, hide frame
    big_ax = fig.add_subplot(111, frameon=False)
    # Hide tick and tick label of the big axis
    big_ax.tick_params(
        labelcolor="none", top=False, bottom=False, left=False, right=False
    )
    big_ax.set_ylabel("Formation energy per atom", labelpad=20, fontsize=16.5)
    big_ax.set_xlabel("Energy above hull", labelpad=20, fontsize=16.5)

    if filename:
        save_plot(figure=fig, filename=filename)

    plt.show()


# %%
def label_dist3(codf, datadf, ehull=False, prop="synth", filename=None):
    plot_df = codf.merge(
        datadf[["material_id", "formation_energy_per_atom", "energy_above_hull"]],
        on="material_id",
    )
    if ehull:
        prop = "stability"
    edf = plot_df[plot_df[prop] == 1]
    tdf = plot_df[plot_df[prop] == 0]
    edf = edf.sort_values("prediction", ascending=False)
    tdf = tdf.sort_values("prediction", ascending=True)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))  # 2x2 subplot grid
    # Define the plots for experimental data
    colors, x, y = density_colors(
        edf.energy_above_hull[edf["prediction"] == 0],
        edf.formation_energy_per_atom[edf["prediction"] == 0],
    )
    scatter1 = axs[0, 0].scatter(
        x,
        y,
        c=colors,
        cmap="viridis",
        norm=LogNorm(),
        alpha=0.7,
    )
    fig.colorbar(scatter1, ax=axs[0, 0], orientation="vertical")
    axs[0, 0].set_title(
        "Experimental Data (Negative Label)", fontsize=13.5, fontweight="bold"
    )

    colors, x, y = density_colors(
        edf.energy_above_hull[edf["prediction"] == 1],
        edf.formation_energy_per_atom[edf["prediction"] == 1],
    )
    scatter2 = axs[1, 0].scatter(
        x,
        y,
        c=colors,
        cmap="viridis",
        norm=LogNorm(),
        alpha=0.7,
    )
    fig.colorbar(scatter2, ax=axs[1, 0], orientation="vertical")
    axs[1, 0].set_title(
        "Experimental Data (Positive Label)", fontsize=13.5, fontweight="bold"
    )

    # # Define the plots for theoretical data
    colors, x, y = density_colors(
        tdf.energy_above_hull[tdf["prediction"] == 0],
        tdf.formation_energy_per_atom[tdf["prediction"] == 0],
    )
    scatter3 = axs[0, 1].scatter(
        x,
        y,
        c=colors,
        cmap="viridis",
        norm=LogNorm(),
        alpha=0.7,
    )
    fig.colorbar(scatter3, ax=axs[0, 1], orientation="vertical")
    axs[0, 1].set_title(
        "Theoretical Data (Negative Label)", fontsize=13.5, fontweight="bold"
    )

    colors, x, y = density_colors(
        tdf.energy_above_hull[tdf["prediction"] == 1],
        tdf.formation_energy_per_atom[tdf["prediction"] == 1],
    )
    scatter4 = axs[1, 1].scatter(
        x,
        y,
        c=colors,
        cmap="viridis",
        norm=LogNorm(),
        alpha=0.7,
    )
    fig.colorbar(scatter4, ax=axs[1, 1], orientation="vertical")
    axs[1, 1].set_title(
        "Theoretical Data (Positive Label)", fontsize=13.5, fontweight="bold"
    )

    # Set common labels
    for ax in axs.flat:
        # ax.set(xlabel='Energy above hull', ylabel='Formation energy per atom')
        ax.label_outer()  # Only show outer labels

    # Adjust subplot params
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    # Add a big axis, hide frame
    big_ax = fig.add_subplot(111, frameon=False)
    # Hide tick and tick label of the big axis
    big_ax.tick_params(
        labelcolor="none", top=False, bottom=False, left=False, right=False
    )
    big_ax.set_ylabel("Formation energy per atom", labelpad=20, fontsize=16.5)
    big_ax.set_xlabel("Energy above hull", labelpad=20, fontsize=16.5)

    if filename:
        save_plot(figure=fig, filename=filename)

    plt.show()


# %%
def label_dist4(
    codf, datadf, pred_col="prediction", ehull=False, prop="synth", filename=None
):
    plot_df = codf.merge(
        datadf[["material_id", "formation_energy_per_atom", "energy_above_hull"]],
        on="material_id",
    )
    if ehull:
        prop = "stability"
    edf = plot_df[plot_df[prop] == 1]
    tdf = plot_df[plot_df[prop] == 0]
    edf = edf.sort_values(pred_col, ascending=False)
    tdf = tdf.sort_values(pred_col, ascending=True)
    # Create a GridSpec with 2 rows and 3 columns
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 0.05], wspace=0.3)
    # Create subplots in the first two columns
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(gs[1, 0], sharex=ax1, sharey=ax1)
    ax4 = fig.add_subplot(gs[1, 1], sharex=ax1, sharey=ax1)
    # Replace axs with the newly created axes
    axs = [[ax1, ax2], [ax3, ax4]]
    for ax in axs[0]:
        for spine in ax.spines.values():
            spine.set_edgecolor("red")
            spine.set_linewidth(2)  # Increase the thickness of the outline
        ax.tick_params(
            colors="red", labelsize=11
        )  # Change the color of the tick labels
    # Change the frame color and thickness of the second row to green
    for ax in axs[1]:
        for spine in ax.spines.values():
            spine.set_edgecolor("green")
            spine.set_linewidth(2)  # Increase the thickness of the outline
        ax.tick_params(
            colors="green", labelsize=11
        )  # Change the color of the tick labels

    all_densities = np.concatenate(
        [
            density_colors(
                edf.energy_above_hull[edf[pred_col] == 0],
                edf.formation_energy_per_atom[edf[pred_col] == 0],
            )[0],
            density_colors(
                edf.energy_above_hull[edf[pred_col] == 1],
                edf.formation_energy_per_atom[edf[pred_col] == 1],
            )[0],
            density_colors(
                tdf.energy_above_hull[tdf[pred_col] == 0],
                tdf.formation_energy_per_atom[tdf[pred_col] == 0],
            )[0],
            density_colors(
                tdf.energy_above_hull[tdf[pred_col] == 1],
                tdf.formation_energy_per_atom[tdf[pred_col] == 1],
            )[0],
        ]
    )
    # Define a common norm object based on the combined range of values
    norm = LogNorm(vmin=all_densities.min(), vmax=all_densities.max())
    # Define the plots for experimental data
    colors, x, y = density_colors(
        edf.energy_above_hull[edf[pred_col] == 0],
        edf.formation_energy_per_atom[edf[pred_col] == 0],
    )
    scatter1 = axs[0][0].scatter(
        x,
        y,
        c=colors,
        cmap="viridis",
        norm=norm,
        alpha=0.7,
    )
    axs[0][0].set_title(
        "Experimental Data", color="blue", fontsize=14, fontweight="bold"
    )

    colors, x, y = density_colors(
        edf.energy_above_hull[edf[pred_col] == 1],
        edf.formation_energy_per_atom[edf[pred_col] == 1],
    )
    scatter2 = axs[1][0].scatter(
        x,
        y,
        c=colors,
        cmap="viridis",
        norm=norm,
        alpha=0.7,
    )
    axs[1][0].set_title(
        "Experimental Data", color="blue", fontsize=14, fontweight="bold"
    )
    # axs[1][0].set_ylabel('Positive Labels', color = 'green',fontsize=14, fontweight = 'bold')

    # # Define the plots for theoretical data
    colors, x, y = density_colors(
        tdf.energy_above_hull[tdf[pred_col] == 0],
        tdf.formation_energy_per_atom[tdf[pred_col] == 0],
    )
    scatter3 = axs[0][1].scatter(
        x,
        y,
        c=colors,
        cmap="viridis",
        norm=norm,
        alpha=0.7,
    )
    axs[0][1].set_title(
        "Theoretical Data", color="#767171", fontsize=14, fontweight="bold"
    )

    ax5 = axs[0][1].twinx()
    ax5.set_frame_on(False)
    ax5.set_ylabel(
        "Negative Labels", color="red", fontsize=14, fontweight="bold", labelpad=20
    )
    ax5.yaxis.set_label_position("right")
    ax5.tick_params(
        axis="y",
        which="both",
        left=False,
        right=False,
        labelleft=False,
        labelright=False,
    )

    axs[0][1].set_ylabel(
        "Negative Labels",
        color="red",
        fontsize=14,
        fontweight="bold",
    )
    axs[0][1].yaxis.set_label_position("right")

    colors, x, y = density_colors(
        tdf.energy_above_hull[tdf[pred_col] == 1],
        tdf.formation_energy_per_atom[tdf[pred_col] == 1],
    )
    scatter4 = axs[1][1].scatter(
        x,
        y,
        c=colors,
        cmap="viridis",
        norm=norm,
        alpha=0.7,
    )
    axs[1][1].set_title(
        "Theoretical Data", color="#767171", fontsize=14, fontweight="bold"
    )
    ax6 = axs[1][1].twinx()
    ax6.set_frame_on(False)
    ax6.set_ylabel(
        "Positive Labels", color="green", fontsize=14, fontweight="bold", labelpad=20
    )
    ax6.yaxis.set_label_position("right")
    ax6.tick_params(
        axis="y",
        which="both",
        left=False,
        right=False,
        labelleft=False,
        labelright=False,
    )
    # Create a single colorbar for all subplots
    cax = fig.add_subplot(gs[:, 2])
    sm = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
    fig.colorbar(sm, cax=cax, orientation="vertical")
    # Set common
    for ax_row in axs:
        for ax in ax_row:
            ax.label_outer()
    # Adjust subplot params
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    # Add a big axis, hide frame
    big_ax = fig.add_subplot(111, frameon=False)
    # Hide tick and tick label of the big axis
    big_ax.tick_params(
        labelcolor="none", top=False, bottom=False, left=False, right=False
    )
    big_ax.set_ylabel("Formation energy per atom (eV)", labelpad=20, fontsize=17.5)
    xlabel = big_ax.set_xlabel("Energy above hull (eV)", labelpad=20, fontsize=17.5)
    xlabel.set_position((0.45, -0.1))  # Adjust these values as needed

    if filename:
        save_plot(figure=fig, filename=filename)

    plt.show()


# %%
def label_dist4_frames(
    codf, datadf, pred_col="prediction", ehull=False, prop="synth", filename=None
):
    plot_df = codf.merge(
        datadf[["material_id", "formation_energy_per_atom", "energy_above_hull"]],
        on="material_id",
    )
    if ehull:
        prop = "stability"

    edf = plot_df[plot_df[prop] == 1]
    tdf = plot_df[plot_df[prop] == 0]
    edf = edf.sort_values(pred_col, ascending=False)
    tdf = tdf.sort_values(pred_col, ascending=True)

    # Create a GridSpec with 2 rows and 3 columns
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 0.05], wspace=0.3)

    # Create subplots in the first two columns
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(gs[1, 0], sharex=ax1, sharey=ax1)
    ax4 = fig.add_subplot(gs[1, 1], sharex=ax1, sharey=ax1)

    axs = [[ax1, ax2], [ax3, ax4]]

    # Define a common norm object based on the combined range of values
    all_densities = np.concatenate(
        [
            density_colors(
                edf.energy_above_hull[edf[pred_col] == 0],
                edf.formation_energy_per_atom[edf[pred_col] == 0],
            )[0],
            density_colors(
                edf.energy_above_hull[edf[pred_col] == 1],
                edf.formation_energy_per_atom[edf[pred_col] == 1],
            )[0],
            density_colors(
                tdf.energy_above_hull[tdf[pred_col] == 0],
                tdf.formation_energy_per_atom[tdf[pred_col] == 0],
            )[0],
            density_colors(
                tdf.energy_above_hull[tdf[pred_col] == 1],
                tdf.formation_energy_per_atom[tdf[pred_col] == 1],
            )[0],
        ]
    )
    norm = LogNorm(vmin=all_densities.min(), vmax=all_densities.max())

    # Experimental data scatter plots
    colors, x, y = density_colors(
        edf.energy_above_hull[edf[pred_col] == 0],
        edf.formation_energy_per_atom[edf[pred_col] == 0],
    )
    scatter1 = axs[0][0].scatter(x, y, c=colors, cmap="viridis", norm=norm, alpha=0.7)
    axs[0][0].set_title(
        "Experimental Data", color="#0072BD", fontsize=14, fontweight="bold"
    )

    colors, x, y = density_colors(
        edf.energy_above_hull[edf[pred_col] == 1],
        edf.formation_energy_per_atom[edf[pred_col] == 1],
    )
    scatter2 = axs[1][0].scatter(x, y, c=colors, cmap="viridis", norm=norm, alpha=0.7)
    axs[1][0].set_title(
        "Experimental Data", color="#0072BD", fontsize=14, fontweight="bold"
    )

    # Theoretical data scatter plots
    colors, x, y = density_colors(
        tdf.energy_above_hull[tdf[pred_col] == 0],
        tdf.formation_energy_per_atom[tdf[pred_col] == 0],
    )
    scatter3 = axs[0][1].scatter(x, y, c=colors, cmap="viridis", norm=norm, alpha=0.7)
    axs[0][1].set_title(
        "Theoretical Data", color="#767171", fontsize=14, fontweight="bold"
    )

    colors, x, y = density_colors(
        tdf.energy_above_hull[tdf[pred_col] == 1],
        tdf.formation_energy_per_atom[tdf[pred_col] == 1],
    )
    scatter4 = axs[1][1].scatter(x, y, c=colors, cmap="viridis", norm=norm, alpha=0.7)
    axs[1][1].set_title(
        "Theoretical Data", color="#767171", fontsize=14, fontweight="bold"
    )

    # Change the spines (plot frame) color of the Experimental Data plots to blue
    for ax in [axs[0][0], axs[1][0]]:  # Only the left plots
        for spine in ax.spines.values():
            spine.set_edgecolor("#0072BD")  # Set the color of the frame
            spine.set_linewidth(2)  # Optionally, make the frame thicker

    # Adjusted frames for Negative and Positive Labels with rounded corners
    rect_negative = FancyBboxPatch(
        (0.14, 0.56),
        0.65,
        0.31,
        transform=fig.transFigure,
        boxstyle="round,pad=0.05",  # "round" style with padding
        edgecolor="red",
        linewidth=3,
        facecolor="none",
    )
    fig.patches.append(rect_negative)

    rect_positive = FancyBboxPatch(
        (0.14, 0.12),
        0.65,
        0.31,
        transform=fig.transFigure,
        boxstyle="round,pad=0.05",  # "round" style with padding
        edgecolor="green",
        linewidth=3,
        facecolor="none",
    )
    fig.patches.append(rect_positive)

    # Add twin axes for Negative and Positive Labels on the right-hand side
    ax5 = axs[0][1].twinx()  # Top-right (Theoretical, Negative Labels)
    ax5.set_frame_on(False)
    ax5.set_ylabel(
        "Negative Labels", color="red", fontsize=14, fontweight="bold", labelpad=30
    )
    ax5.yaxis.set_label_position("right")
    ax5.tick_params(
        axis="y",
        which="both",
        left=False,
        right=False,
        labelleft=False,
        labelright=False,
    )

    ax6 = axs[1][1].twinx()  # Bottom-right (Theoretical, Positive Labels)
    ax6.set_frame_on(False)
    ax6.set_ylabel(
        "Positive Labels", color="green", fontsize=14, fontweight="bold", labelpad=30
    )
    ax6.yaxis.set_label_position("right")
    ax6.tick_params(
        axis="y",
        which="both",
        left=False,
        right=False,
        labelleft=False,
        labelright=False,
    )

    # Create a single colorbar for all subplots
    cax = fig.add_subplot(gs[:, 2])
    sm = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
    fig.colorbar(sm, cax=cax, orientation="vertical")

    # Set common labels
    big_ax = fig.add_subplot(111, frameon=False)
    big_ax.tick_params(
        labelcolor="none", top=False, bottom=False, left=False, right=False
    )
    big_ax.set_ylabel("Formation energy per atom (eV)", labelpad=20, fontsize=17.5)
    xlabel = big_ax.set_xlabel("Energy above hull (eV)", labelpad=20, fontsize=17.5)
    xlabel.set_position((0.45, -0.1))  # Adjust these values as needed

    # Adjust subplot spacing with `hspace` to create more space between the rows
    plt.subplots_adjust(hspace=0.4)

    # Save the plot
    if filename:
        fig.savefig(filename, dpi=300, bbox_inches="tight")

    plt.show()


# %%
# %%
def midlabel_dist(
    codf, datadf, ehull=False, prop="synth", figtitle=None, filename=None
):
    datadf["original_index"] = datadf.index

    plot_df = codf.merge(
        datadf[
            [
                "material_id",
                "formation_energy_per_atom",
                "energy_above_hull",
                "original_index",
            ]
        ],
        on="material_id",
    )
    plot_df.set_index("original_index", inplace=True, drop=True)
    if ehull:
        prop = "stability"
        LOTestPath = "../clean_data/alignn0_stability/leaveout_test_id.txt"
    else:
        LOTestPath = "../clean_data/alignn0_synth/leaveout_test_id.txt"
    edf = plot_df[plot_df[prop] == 1]
    tdf = plot_df[plot_df[prop] == 0]
    with open(LOTestPath, "r") as ff:
        id_LOtest = [int(line.strip()) for line in ff]
    LO_testSet = edf.loc[id_LOtest]
    testSet = edf.drop(index=LO_testSet.index)
    LO_true_positive_rate = LO_testSet.prediction.mean()
    true_positive_rate = testSet.prediction.mean()
    # tpr_mean = np.mean([LO_true_positive_rate, true_positive_rate])
    unlabeled_synth_frac = tdf.prediction.mean()
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 8))
    if "predScore" in edf.columns:
        prob = "predScore"
    else:
        prob = "avg_prediction"
    n, bins, patches = ax1.hist(edf[prob], bins=40)
    max_height = n.max()
    ax1.set_title(
        "Synthesizability Score for Experimental Data", fontsize=13.5, fontweight="bold"
    )
    ax1.vlines(0.5, 0, max_height, "r", linewidth=3)
    ax1.arrow(
        0.5,
        max_height * 0.7,
        0.35,
        0,
        linewidth=3,
        color="r",
        length_includes_head=True,
        head_width=max_height * 0.08,
        head_length=0.05,
    )
    ax1.text(
        0.55,
        max_height * 0.6,
        "{:.1f}% Recall".format(true_positive_rate * 100),
        fontsize=14,
    )
    ax1.text(
        0.55,
        max_height * 0.4,
        "{:.1f}% Leave-out Recall".format(LO_true_positive_rate * 100),
        fontsize=14,
    )
    # ax1.text(.55,max_height*.4, '{:.1f}% true- \npositive rate'.format(tpr_mean*100), fontsize = 15);

    plt.xlabel("Predicted Probability of Synthesizability", fontsize=14)
    # plt.xlabel('Synthesizability Score', fontsize=14)

    n, bins, patches = ax2.hist(tdf[prob], bins=40)
    max_height = n.max()
    ax2.vlines(0.5, 0, max_height, "r", linewidth=3)
    ax2.arrow(
        0.5,
        max_height * 0.7,
        0.35,
        0,
        linewidth=3,
        color="r",
        length_includes_head=True,
        head_width=max_height * 0.08,
        head_length=0.05,
    )
    ax2.set_title(
        "Synthesizability Score for Theoretical Data", fontsize=13.5, fontweight="bold"
    )
    ax2.text(
        0.55,
        max_height * 0.4,
        "{:.1f}% predicted \n synthesizable".format(unlabeled_synth_frac * 100),
        fontsize=14,
    )
    if figtitle:
        fig.suptitle(figtitle, fontsize=25, y=1.02)
    if filename:
        save_plot(figure=fig, filename=filename)
    plt.show()


# %%
# midlabel_dist(codf, df)


# %%
# with open(f"leaveout_test_material_id.txt", "w") as f:
#     for test_material_id in mdf.loc[id_LOtest].material_id:
#         f.write(str(test_material_id) + "\n")
# #mdf is synthdf or schnet0 or coSchnet...
# %%
def final_labels(
    plot_df, ehull=False, prop="synth", figtitle=None, filename=None, threshold=0.5
):
    edf = plot_df[plot_df[prop] == 1]
    tdf = plot_df[plot_df[prop] == 0]
    # prob = "prediction"
    # prob_col = "synth_avg"
    prob_col = "synth_preds"

    LOTestPath = "leaveout_test_material_id.txt"
    # edf = plot_df[plot_df[prop] == 1]
    # tdf = plot_df[plot_df[prop] == 0]
    with open(LOTestPath, "r") as ff:
        id_LOtest_mat = [line.strip() for line in ff]
        # id_LOtest = [int(line.strip()) for line in ff]
    edf.set_index("material_id", inplace=True)
    LO_testSet = edf.loc[id_LOtest_mat]
    testSet = edf.drop(index=LO_testSet.index)
    LO_true_positive_rate = LO_testSet[prob_col].mean()
    true_positive_rate = testSet[prob_col].mean()
    # LO_true_positive_rate = LO_testSet.prediction.mean()
    # true_positive_rate = testSet.prediction.mean()
    print(
        f"LO_true_positive_rate is {LO_true_positive_rate} and True_positive_rate is {true_positive_rate}"
    )
    # tpr_mean = np.mean([LO_true_positive_rate, true_positive_rate])
    unlabeled_synth_frac = tdf[prob_col].mean()
    # unlabeled_synth_frac = tdf.prediction.mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 8))
    if threshold <= 0.5:
        # Position the text to the right of the vline
        text_halign = "left"
        text_offset = 0.03
    else:
        # Position the text to the left of the vline
        text_halign = "right"
        text_offset = -0.03
    n, bins, patches = ax1.hist(edf[prob_col], bins=40)
    max_height = n.max()
    ax1.set_title(
        "Synthesizability Prediction for Experimental Data",
        fontsize=13.5,
        fontweight="bold",
    )
    ax1.vlines(threshold, 0, max_height, "r", linewidth=3)
    ax1.arrow(
        threshold,
        max_height * 0.7,
        0.35,
        0,
        linewidth=3,
        color="r",
        length_includes_head=True,
        head_width=max_height * 0.08,
        head_length=0.05,
    )
    # ax1.text(threshold+.03,max_height*.6, '{:.1f}% Recall'.format(true_positive_rate*100), fontsize = 14);
    ax1.text(
        threshold + text_offset,
        max_height * 0.6,
        "{:.1f}% Recall".format(true_positive_rate * 100),
        fontsize=14,
        ha=text_halign,
    )
    ax1.text(
        threshold + text_offset,
        max_height * 0.4,
        "{:.1f}% Leave-out Recall".format(LO_true_positive_rate * 100),
        fontsize=14,
        ha=text_halign,
    )
    # ax1.text(.55,max_height*.4, '{:.1f}% true- \npositive rate'.format(true_positive_rate*100), fontsize = 15);

    plt.xlabel("Predicted Classes of Synthesizability", fontsize=14, labelpad=10)

    n, bins, patches = ax2.hist(tdf[prob_col], bins=40)
    max_height = n.max()
    ax2.vlines(threshold, 0, max_height, "r", linewidth=3)
    ax2.arrow(
        threshold,
        max_height * 0.7,
        0.35,
        0,
        linewidth=3,
        color="r",
        length_includes_head=True,
        head_width=max_height * 0.08,
        head_length=0.05,
    )
    ax2.set_title(
        "Synthesizability Prediction for Theoretical Data",
        fontsize=13.5,
        fontweight="bold",
    )
    ax2.text(
        threshold + text_offset,
        max_height * 0.4,
        "{:.1f}% predicted \n synthesizable".format(unlabeled_synth_frac * 100),
        fontsize=14,
        ha=text_halign,
    )
    if figtitle:
        fig.suptitle(figtitle, fontsize=25, y=1.02)
    if filename:
        save_plot(figure=fig, filename=filename)
    plt.show()


# %%
def scatter_hm_final(proplab, datadf, prop="synth", filename=None):
    proplab = proplab.merge(
        datadf[["material_id", "formation_energy_per_atom", "energy_above_hull"]],
        on="material_id",
    )
    proplab = proplab.dropna()
    colors, x, y = density_colors(proplab[f"{prop}_avg"], proplab.energy_above_hull)
    plt.figure(figsize=(10, 6))
    # Plot first scatter plot
    plt.scatter(x, y, c=colors, cmap="viridis", norm=LogNorm(), alpha=0.7, s=20)
    plt.colorbar()
    plt.xlabel("Predicted Probability of Synthesizability", fontsize=15, labelpad=10)
    plt.ylabel("Energy Above Hull (eV)", fontsize=15, labelpad=10)
    plt.title("Density Scatter Plot", fontsize=17)
    if filename:
        save_plot(plt.gcf(), filename)
    plt.show()


# %%
def scatter_hm_final_frac(proplab, datadf, prop="synth", filename=None):
    proplab = proplab.merge(
        datadf[["material_id", "formation_energy_per_atom", "energy_above_hull"]],
        on="material_id",
    )
    proplab = proplab.dropna()
    tdf = proplab[proplab[prop] == 0]
    colors, x, y, heatmap, xedges, yedges = density_colors_with_contours(
        tdf[f"{prop}_avg"], tdf.energy_above_hull
    )

    # Calculate fractions
    total_points = len(x)
    lower_left = sum((x_val < 0.5) and (y_val <= 1) for x_val, y_val in zip(x, y))
    lower_right = sum((x_val >= 0.5) and (y_val <= 1) for x_val, y_val in zip(x, y))
    upper_left = sum((x_val < 0.5) and (y_val > 1) for x_val, y_val in zip(x, y))
    upper_right = sum((x_val >= 0.5) and (y_val > 1) for x_val, y_val in zip(x, y))

    fractions = {
        "lower_left": lower_left / total_points,
        "lower_right": lower_right / total_points,
        "upper_left": upper_left / total_points,
        "upper_right": upper_right / total_points,
    }

    plt.figure(figsize=(10, 6))
    # Plot first scatter plot
    plt.scatter(x, y, c=colors, cmap="viridis", norm=LogNorm(), alpha=0.7, s=20)
    plt.colorbar()

    # Contour plot
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])  # Convert bin edges to grid
    contour_levels = [
        np.percentile(heatmap[heatmap > 0], 70)
    ]  # 80th percentile as an example
    plt.contour(X, Y, heatmap.T, levels=contour_levels, colors="black", linewidths=4)

    plt.xlabel("Predicted Probability of Synthesizability", fontsize=15, labelpad=10)
    plt.ylabel("Energy Above Hull (eV)", fontsize=15, labelpad=10)
    plt.title("Density Scatter Plot", fontsize=17)

    # Add dividing lines
    plt.axvline(x=0.5, color="r", linestyle="--", linewidth=2)
    plt.axhline(y=1, color="r", linestyle="-", linewidth=2)

    # Add text annotations for fractions
    plt.text(
        0.25,
        0.70,
        f"{fractions['upper_left']:.1%}",
        horizontalalignment="center",
        verticalalignment="center",
        transform=plt.gca().transAxes,
        color="DodgerBlue",
        fontsize=25,
        weight="bold",
    )
    plt.text(
        0.75,
        0.70,
        f"{fractions['upper_right']:.2%}",
        horizontalalignment="center",
        verticalalignment="center",
        transform=plt.gca().transAxes,
        color="DodgerBlue",
        fontsize=25,
        weight="bold",
    )
    plt.text(
        0.25,
        0.062,
        f"{fractions['lower_left']:.1%}",
        horizontalalignment="center",
        verticalalignment="center",
        transform=plt.gca().transAxes,
        color="red",
        fontsize=25,
        weight="bold",
    )
    plt.text(
        0.75,
        0.062,
        f"{fractions['lower_right']:.1%}",
        horizontalalignment="center",
        verticalalignment="center",
        transform=plt.gca().transAxes,
        color="red",
        fontsize=25,
        weight="bold",
    )

    if filename:
        save_plot(plt.gcf(), filename)
    plt.show()


# %%
def heatmap_final(codf, datadf, prop="synth", filename=None):
    plot_df = codf.merge(
        datadf[["material_id", "formation_energy_per_atom", "energy_above_hull"]],
        on="material_id",
    )
    plot_df = plot_df.dropna()
    x = plot_df.predScore
    y = plot_df.energy_above_hull
    # x = x[~np.isnan(x)]
    # y = y[~np.isnan(y)]

    xzoomed = plot_df[f"{prop}_avg"][plot_df.energy_above_hull <= 1]
    yzoomed = plot_df.energy_above_hull[plot_df.energy_above_hull <= 1]
    xzoomed = xzoomed[~np.isnan(xzoomed)]
    yzoomed = yzoomed[~np.isnan(yzoomed)]

    # Calculate the bins for the full dataset
    hist_full, xedges, yedges = np.histogram2d(x, y, bins=(50, 50))

    # Use the full data set to find the common vmin and vmax
    vmin = 1  # Since you use cmin=1, we start the colorbar at 1
    vmax = hist_full.max()  # The max count from the full data set

    # First plot
    plt.hist2d(
        x, y, bins=(50, 50), cmap="viridis", norm=LogNorm(vmin=vmin, vmax=vmax), cmin=1
    )
    cbar = plt.colorbar()
    plt.xlabel("Predicted Probability of Synthesizability")
    plt.ylabel("energy_above_hull")
    plt.title("Density Heatmap")
    if filename:
        save_plot(plt.gcf(), filename)
    plt.show()


# %%
# %%
# %%
"""
schnet0 = pd.read_pickle('../results/synth/schnet0.pkl')
alignn0 = pd.read_pickle('../results/synth/alignn0.pkl')

coschnet1 = pd.read_pickle('../results/synth/coSchnet1.pkl')
coalignn1 = pd.read_pickle('../results/synth/coAlignn1.pkl')

coschnet2 = pd.read_pickle('../results/synth/coSchnet2.pkl')
coalignn2 = pd.read_pickle('../results/synth/coAlignn2.pkl')

coschnet3 = pd.read_pickle('../results/synth/coSchnet3.pkl')
coalignn3 = pd.read_pickle('../results/synth/coAlignn3.pkl')

synthlab = pd.read_pickle(os.path.join(
    os.path.dirname(__file__),'../results/synth/synth_labels_2'))
# %%

midlabel_dist(schnet0, df, figtitle="Iteration '0' with SchNet", 
              filename="schnet0_prop_dist.png")
midlabel_dist(alignn0, df, figtitle="Iteration '0' with ALIGNN", 
              filename="alignn0_prop_dist.png")
            # )            
# %%
midlabel_dist(coalignn1, df, figtitle="Iteration '1' with ALIGNN", 
              filename="coalignn1_prop_dist.png")
midlabel_dist(coschnet1, df, figtitle="Iteration '1' with SchNet", 
              filename="coschnet1_prop_dist.png")
# %%
midlabel_dist(coalignn2, df, figtitle="Iteration '2' with ALIGNN", 
              filename="coalignn2_prop_dist.png")
midlabel_dist(coschnet2, df, figtitle="Iteration '2' with SchNet", 
              filename="coschnet2_prop_dist.png")
# %%
midlabel_dist(coalignn3, df, figtitle="Iteration '3' with ALIGNN", 
              filename="coalignn3_prop_dist.png")
midlabel_dist(coschnet3, df, figtitle="Iteration '3' with SchNet", 
              filename="coschnet3_prop_dist.png")
            
# %%
# midlabel_dist(schnet0, df, figtitle=None, 
#               filename="schnet0_prop_dist_nt.png")
# midlabel_dist(alignn0, df, figtitle=None, 
#               filename="alignn0_prop_dist_nt.png")
# midlabel_dist(coalignn3, df, figtitle=None, 
#               filename="coalignn3_prop_dist_nt.png")
# midlabel_dist(coschnet3, df, figtitle=None, 
#               filename="coschnet3_prop_dist_nt.png")

# %%
# finaldf = pd.read_pickle(os.path.join(
#     os.path.dirname(__file__),'../../predict_target/final_df'))
# %%
scatter_hm_final_frac(synthlab,df, prop=prop, filename='final_sctter_hm_frac_it2.pdf')
# %%
final_labels(synthlab, figtitle="Label Distribution After Averaging")#, filename='final_label_dist_it2.png')    
# %%
# label_dist4(codf, datadf, pred_col = 'prediction' ,ehull=False,prop = prop, filename=None)
label_dist4(synthlab, df, pred_col='synth_preds', filename='final_label_dist_it2.png')
# %%
synthlab_t0 = pd.read_pickle(os.path.join(
    os.path.dirname(__file__),'../results/synth/synth_labels_2_threshold_0_0'))
synthlab_t1 = pd.read_pickle(os.path.join(
    os.path.dirname(__file__),'../results/synth/synth_labels_2_threshold_1_0'))
synthlab_t25 = pd.read_pickle(os.path.join(
    os.path.dirname(__file__),'../results/synth/synth_labels_2_threshold_0_25'))
synthlab_t75 = pd.read_pickle(os.path.join(
    os.path.dirname(__file__),'../results/synth/synth_labels_2_threshold_0_75'))
# %%
final_labels(synthlab_t75, figtitle="Label Distribution with 0.75 Threshold",
             filename='final_label_dist_it2_75.png', threshold=0.75)
final_labels(synthlab_t25, figtitle="Label Distribution with 0.25 Threshold",
             filename='final_label_dist_it2_25.png', threshold=0.25)
# %%
# try this with the condition of where the threshold is:
#     # Place text on the plot so that it ends at `threshold`
# ax1.text(threshold, max_height*0.7, text, ha='right', va='center', color='black')


# %%
"""
# %%

# import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def label_dist5(
    codf, datadf, pred_col="prediction", ehull=False, prop="synth", filename=None
):
    plot_df = codf.merge(
        datadf[["material_id", "formation_energy_per_atom", "energy_above_hull"]],
        on="material_id",
    )
    if ehull:
        prop = "stability"

    edf = plot_df[plot_df[prop] == 1]
    tdf = plot_df[plot_df[prop] == 0]
    edf = edf.sort_values(pred_col, ascending=False)
    tdf = tdf.sort_values(pred_col, ascending=True)

    # Create a GridSpec with 2 rows and 3 columns
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 0.05], wspace=0.3)

    # Create subplots in the first two columns
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(gs[1, 0], sharex=ax1, sharey=ax1)
    ax4 = fig.add_subplot(gs[1, 1], sharex=ax1, sharey=ax1)

    axs = [[ax1, ax2], [ax3, ax4]]

    # Default black edges for scatter plots
    for ax_row in axs:
        for ax in ax_row:
            for spine in ax.spines.values():
                spine.set_edgecolor("black")
                spine.set_linewidth(1)
            ax.tick_params(colors="black", labelsize=11)

    # Create density scatter plots (this part is unchanged)
    all_densities = np.concatenate(
        [
            density_colors(
                edf.energy_above_hull[edf[pred_col] == 0],
                edf.formation_energy_per_atom[edf[pred_col] == 0],
            )[0],
            density_colors(
                edf.energy_above_hull[edf[pred_col] == 1],
                edf.formation_energy_per_atom[edf[pred_col] == 1],
            )[0],
            density_colors(
                tdf.energy_above_hull[tdf[pred_col] == 0],
                tdf.formation_energy_per_atom[tdf[pred_col] == 0],
            )[0],
            density_colors(
                tdf.energy_above_hull[tdf[pred_col] == 1],
                tdf.formation_energy_per_atom[tdf[pred_col] == 1],
            )[0],
        ]
    )
    norm = LogNorm(vmin=all_densities.min(), vmax=all_densities.max())

    # Experimental data scatter plots
    colors, x, y = density_colors(
        edf.energy_above_hull[edf[pred_col] == 0],
        edf.formation_energy_per_atom[edf[pred_col] == 0],
    )
    scatter1 = axs[0][0].scatter(x, y, c=colors, cmap="viridis", norm=norm, alpha=0.7)

    colors, x, y = density_colors(
        edf.energy_above_hull[edf[pred_col] == 1],
        edf.formation_energy_per_atom[edf[pred_col] == 1],
    )
    scatter2 = axs[1][0].scatter(x, y, c=colors, cmap="viridis", norm=norm, alpha=0.7)

    # Theoretical data scatter plots
    colors, x, y = density_colors(
        tdf.energy_above_hull[tdf[pred_col] == 0],
        tdf.formation_energy_per_atom[tdf[pred_col] == 0],
    )
    scatter3 = axs[0][1].scatter(x, y, c=colors, cmap="viridis", norm=norm, alpha=0.7)

    colors, x, y = density_colors(
        tdf.energy_above_hull[tdf[pred_col] == 1],
        tdf.formation_energy_per_atom[tdf[pred_col] == 1],
    )
    scatter4 = axs[1][1].scatter(x, y, c=colors, cmap="viridis", norm=norm, alpha=0.7)

    # Add a big frame around the experimental data
    rect_exp = Rectangle(
        (0.1, 0.1),
        0.4,
        0.85,
        transform=fig.transFigure,
        color="blue",
        fill=False,
        linewidth=3,
    )
    fig.patches.append(rect_exp)
    fig.text(
        0.3,
        0.95,
        "Experimental Data",
        fontsize=16,
        fontweight="bold",
        color="blue",
        ha="center",
    )

    # Add a big frame around the theoretical data
    rect_theo = Rectangle(
        (0.52, 0.1),
        0.4,
        0.85,
        transform=fig.transFigure,
        color="#767171",
        fill=False,
        linewidth=3,
    )
    fig.patches.append(rect_theo)
    fig.text(
        0.72,
        0.95,
        "Theoretical Data",
        fontsize=16,
        fontweight="bold",
        color="#767171",
        ha="center",
    )

    # Create a single colorbar for all subplots
    cax = fig.add_subplot(gs[:, 2])
    sm = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
    fig.colorbar(sm, cax=cax, orientation="vertical")

    # Set common labels
    big_ax = fig.add_subplot(111, frameon=False)
    big_ax.tick_params(
        labelcolor="none", top=False, bottom=False, left=False, right=False
    )
    big_ax.set_ylabel("Formation energy per atom (eV)", labelpad=20, fontsize=17.5)
    big_ax.set_xlabel("Energy above hull (eV)", labelpad=20, fontsize=17.5)

    # Save the plot
    if filename:
        fig.savefig(filename, dpi=300, bbox_inches="tight")

    plt.show()


# %%
