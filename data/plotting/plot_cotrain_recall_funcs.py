# %%
import numpy as np
import matplotlib.pyplot as plt


# %%
def plot_recall_stability(res_df, y_lims=None, title="Recall Rate"):
    x = range(len(res_df.exper))  # Create a range of numbers
    y = 100 * res_df.true_positive_rate  # Values for y-axis
    y2 = 100 * res_df.LO_true_positive_rate  # Create a range of numbers

    plt.figure(figsize=(5, 3))  # Adjust the figure size as needed
    plt.plot(x, y, "--o", label="Averaged Recall")  # Plot with markers for each point
    plt.plot(x, y2, "--o", label="Leave-out Recall")  # Plot with markers for each point
    # try:
    y3 = 100 * res_df.GT_true_positive_rate  # Create a range of numbers
    plt.plot(
        x, y3, "-*", markersize=10, label="Ground-truth Recall"
    )  # Plot with larger markers for each point
    # except:
    #     print('No Ground Truth data here.')
    #     pass
    y_min = np.minimum(y, y2)
    y_max = np.maximum(y, y2)
    # Fill the area between y_min and y_max
    plt.fill_between(x, y_min, y_max, color="gray", alpha=0.5)
    # Set the x-axis tick labels to res_df.exper, rotated for readability
    plt.xticks(ticks=x, labels=res_df.exper, rotation=45, fontsize=11)
    # Set labels for axes
    plt.xlabel("Experiments", fontsize=13, labelpad=10)
    plt.ylabel("Recall (%)", fontsize=15, labelpad=10)
    plt.grid(which="major", linestyle="-", linewidth="0.5", color="gray")
    if y_lims:
        plt.ylim(y_lims)
    # Optionally, you can add a title
    plt.title(title, fontsize=15)
    plt.legend(fontsize=13, loc="lower right")
    if title:
        plt.savefig(
            f"{title.replace(' ', '_')}.pdf",
            dpi=350,
            transparent=True,
            bbox_inches="tight",
        )
    else:
        plt.show()


# %%
def plot_recall_synth(res_df, y_lims=None, title="Recall Rate"):
    x = range(len(res_df.exper))  # Create a range of numbers
    y = 100 * res_df.true_positive_rate  # Values for y-axis
    y2 = 100 * res_df.LO_true_positive_rate  # Create a range of numbers
    y_ppr = 100 * res_df.predicted_positive_rate  # Create a range of numbers

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 7), sharex=True)

    ax1.plot(x, y, "--o", label="Averaged Recall")
    ax1.plot(x, y2, "--o", label="Leave-out Recall")
    ax1.set_ylabel("Recall (%)", fontsize=15)
    ax1.legend(loc="best", fontsize=15)
    ax1.grid(True)
    y_min = np.minimum(y, y2)
    y_max = np.maximum(y, y2)
    ax1.fill_between(x, y_min, y_max, color="gray", alpha=0.4)
    ax1.tick_params(axis="y", labelsize=15)

    ax2.plot(x, y_ppr, "--o", label="Predicted Positive Rate", color="magenta")
    ax2.set_xlabel("Experiments", fontsize=17)
    ax2.set_ylabel("Predicted Positive Rate (%)", fontsize=15)
    ax2.legend(loc="best", fontsize=15)
    ax2.tick_params(axis="y", labelsize=15)
    ax2.grid(True)

    plt.xticks(ticks=x, labels=res_df.exper, rotation=45, fontsize=15)

    if y_lims:
        ax1.set_ylim(y_lims[0])  # Apply the first set of limits to the first subplot
        ax2.set_ylim(y_lims[1])  # Apply the second set of limits to the second subplot

    fig.suptitle(title, fontsize=20)
    plt.tight_layout(pad=2.0)  # Increase padding to prevent overlap with the title

    if title:
        plt.savefig(
            f"{title.replace(' ', '_')}.pdf",
            dpi=350,
            transparent=True,
            bbox_inches="tight",
        )
    else:
        plt.show()


# %%
def df_by_series(serdf):
    df1 = serdf[serdf.exper.isin(["SchNet0", "coAlignn1", "coSchNet2", "coAlignn3"])]
    df2 = serdf[~serdf.exper.isin(["SchNet0", "coAlignn1", "coSchNet2", "coAlignn3"])]
    return df1, df2


def synth_y_margin(mardf):
    recall_min = 100 * min(
        mardf.true_positive_rate.min(),
        mardf.LO_true_positive_rate.min(),
    )
    recall_max = 100 * max(
        mardf.true_positive_rate.max(),
        mardf.LO_true_positive_rate.max(),
    )
    recall_plot_margin = (recall_max - recall_min) * 0.08
    recall_y_lims = (recall_min - recall_plot_margin, recall_max + recall_plot_margin)

    ppr_min = 100 * min(
        mardf.predicted_positive_rate.min(), mardf.predicted_positive_rate.min()
    )
    ppr_max = 100 * max(
        mardf.predicted_positive_rate.max(), mardf.predicted_positive_rate.max()
    )
    ppr_plot_margin = (ppr_max - ppr_min) * 0.08
    ppr_y_lims = (ppr_min - ppr_plot_margin, ppr_max + ppr_plot_margin)
    return recall_y_lims, ppr_y_lims


def stability_y_margin(mardf):
    recall_min = 100 * min(
        mardf.true_positive_rate.min(),
        mardf.LO_true_positive_rate.min(),
        mardf.GT_true_positive_rate.min(),
    )
    recall_max = 100 * max(
        mardf.true_positive_rate.max(),
        mardf.LO_true_positive_rate.max(),
        mardf.GT_true_positive_rate.max(),
    )
    recall_plot_margin = (recall_max - recall_min) * 0.08
    recall_y_lims = (recall_min - recall_plot_margin, recall_max + recall_plot_margin)

    return recall_y_lims
