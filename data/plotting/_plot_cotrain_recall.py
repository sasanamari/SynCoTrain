# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import os
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import Bbox
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# from brokenaxes import brokenaxes
# import seaborn as sns
# %%
# df = pd.read_csv("../results/stability015/results_plot.csv")
df = pd.read_csv("../results/synth/results.csv")
df015 = pd.read_csv("../results/stability015/results.csv")
df = df.dropna()
df.reset_index(drop=True, inplace=True)
df['exper'].replace({'schnet0': 'SchNet0', 'alignn0': 'Alignn0',
                     'coSchnet2': 'coSchNet2', 'coSchnet3': 'coSchNet3'}, inplace=True)

#nicer plotting labels
# %%
def plot_recall_stability(res_df, y_lims = None, title = 'Recall Rate'):
    x = range(len(res_df.exper))  # Create a range of numbers
    y = res_df.true_positive_rate  # Values for y-axis
    y2 = res_df.LO_true_positive_rate  # Create a range of numbers

    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    plt.plot(x, y, '--o',label = "Averaged Recall")  # Plot with markers for each point
    plt.plot(x, y2, '--o',label = "Leave-out Recall")  # Plot with markers for each point
    # try:
    y3 = res_df.GT_true_positive_rate  # Create a range of numbers
    plt.plot(x, y3, '-*', markersize=10, label="Ground-truth Recall")  # Plot with larger markers for each point
    # except:
    #     print('No Ground Truth data here.')
    #     pass
    y_min = np.minimum(y, y2)
    y_max = np.maximum(y, y2)
    # Fill the area between y_min and y_max
    plt.fill_between(x, y_min, y_max, color='gray', alpha=0.5)
    # Set the x-axis tick labels to res_df.exper, rotated for readability
    plt.xticks(ticks=x, labels=res_df.exper, rotation=45, fontsize =11)
    # Set labels for axes
    plt.xlabel('Experiments', fontsize=13, labelpad=10)
    plt.ylabel('Recall (%)', fontsize=15, labelpad=10)
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
    if y_lims:
        plt.ylim(y_lims)
    # Optionally, you can add a title
    plt.title(title, fontsize=15)
    plt.legend(fontsize = 13, loc='lower right')
    if title:
        plt.savefig(f"{title.replace(' ', '_')}.png", dpi=350, transparent=True, bbox_inches='tight')
    plt.show()
# %%
def plot_recall_synth(res_df, y_lims = None,title = 'Recall Rate'):
    x = range(len(res_df.exper))  # Create a range of numbers
    y = 100*res_df.true_positive_rate  # Values for y-axis
    y2 = 100*res_df.LO_true_positive_rate  # Create a range of numbers
    y_ppr = 100*res_df.predicted_positive_rate  # Create a range of numbers
    # Creating a figure and two subplots, sharing the x-axis
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 12), sharex=True)
    # Plot y and y2 in the first subplot
    ax1.plot(x, y, '--o', label='Averaged Recall')
    ax1.plot(x, y2, '--o', label='Leave-out Recall')
    # Add labels, grid and legend to the first subplot
    ax1.set_ylabel('Recall (%)', fontsize=15)
    ax1.legend(loc='best')
    ax1.grid(True)
    y_min = np.minimum(y, y2)
    y_max = np.maximum(y, y2)
    # Fill the area between y_min and y_max
    ax1.legend(loc='best', fontsize=16)
    ax1.fill_between(x, y_min, y_max, color='gray', alpha=0.4)
    ax1.tick_params(axis='y', labelsize=15)
    # Plot y_ppr in the second subplot
    ax2.plot(x, y_ppr, '--o', label='Predicted Positive Rate', color='magenta')
    # Add labels, grid and legend to the second subplot
    ax2.set_xlabel('Experiments', fontsize=17)
    ax2.set_ylabel('Predicted Positive Rate (%)', fontsize=15)
    ax2.legend(loc='best', fontsize=15)
    ax2.tick_params(axis='y', labelsize=15)
    ax2.grid(True)
    # Set the x-axis tick labels to res_df.exper, rotated for readability
    plt.xticks(ticks=x, labels=res_df.exper, rotation=45, fontsize =15)
    if y_lims:
        plt.ylim(y_lims)
    # plt.title(title, fontsize=15)
    fig.suptitle(title, fontsize=20)
    # Adjust the spacing between the plots
    plt.tight_layout()

    if title:
        plt.savefig(f"{title.replace(' ', '_')}.png", dpi=350, transparent=True, bbox_inches='tight')
    plt.show()    
# %%
def df_by_series(serdf):
    df1 = serdf[serdf.exper.isin(['SchNet0', 'coAlignn1', 'coSchNet2', 'coAlignn3'])]
    df2 = serdf[~serdf.exper.isin(['SchNet0', 'coAlignn1', 'coSchNet2', 'coAlignn3'])]
    return df1, df2
def recall_y_margin(mardf):
    global_y_min = min(mardf.true_positive_rate.min(), mardf.LO_true_positive_rate.min())
    global_y_max = max(mardf.true_positive_rate.max(), mardf.LO_true_positive_rate.max())
    plot_margin = (global_y_max-global_y_min)*0.08
    y_lims = (global_y_min - plot_margin, global_y_max + plot_margin)
    return y_lims
    
# %%
plot_recall_synth(df_by_series(df)[0],title= "SchNet0 Series Synthesizability Prediction Rates")
plot_recall_synth(df_by_series(df)[1],title= "ALIGNN0 Series Synthesizability Prediction Rates")
# %%
plot_recall_stability(df_by_series(df015)[0],title= "SchNet0 Series Stability Recall")
plot_recall_stability(df_by_series(df015)[1],title= "ALIGNN0 Series Stability Recall")

# %%
