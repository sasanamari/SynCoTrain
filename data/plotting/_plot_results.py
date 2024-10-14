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
from brokenaxes import brokenaxes
# import seaborn as sns
# %%
# df = pd.read_csv("../results/stability015/results_plot.csv")
df = pd.read_csv("../results/synth/results_plot_new_names.csv")
df015 = pd.read_csv("../results/stability015/results_plot_new_names.csv")
df = df.dropna()

# %%
# Assuming df015 is your DataFrame
x = range(len(df015.exper))  # Create a range of numbers
y = df015.true_positive_rate  # Values for y-axis
y2 = df015.LO_true_positive_rate  # Create a range of numbers
y3 = df015.GT_true_positive_rate  # Create a range of numbers

plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
plt.plot(x, y, '--o',label = "TPR")  # Plot with markers for each point
plt.plot(x, y2, '--o',label = "LOTPR")  # Plot with markers for each point
plt.plot(x, y3, '-o',label = "GTTPR")  # Plot with markers for each point
y_min = np.minimum(y, y2)
y_max = np.maximum(y, y2)
# Fill the area between y_min and y_max
plt.fill_between(x, y_min, y_max, color='gray', alpha=0.5)
# Set the x-axis tick labels to df.exper, rotated for readability
plt.xticks(ticks=x, labels=df.exper, rotation=45, fontsize =12)
# Set labels for axes
plt.xlabel('Experiment', fontsize=15, labelpad=10)
# plt.xlabel('Experiment')
plt.ylabel('True Positive Rate', fontsize=15, labelpad=10)
plt.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
# Optionally, you can add a title
plt.title('True Positive Rate Over Iterations', fontsize=15)
plt.legend(fontsize = 15)
# plt.savefig("tpr_GT_iterations.png", dpi=350, transparent=True, bbox_inches='tight')
plt.show()
# %%
def plot_recall_iter(res_df, y_lims = None, title = 'Recall Rate'):
    x = range(len(res_df.exper))  # Create a range of numbers
    y = res_df.true_positive_rate  # Values for y-axis
    y2 = res_df.LO_true_positive_rate  # Create a range of numbers

    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    plt.plot(x, y, '--o',label = "Recall")  # Plot with markers for each point
    plt.plot(x, y2, '--o',label = "Leave-out Recall")  # Plot with markers for each point
    try:
        y3 = res_df.GT_true_positive_rate  # Create a range of numbers
        plt.plot(x, y3, '-*', markersize=10, label="Ground-truth Recall")  # Plot with larger markers for each point
    except:
        print('No Ground Truth data here.')
        pass
    y_min = np.minimum(y, y2)
    y_max = np.maximum(y, y2)
    # Fill the area between y_min and y_max
    plt.fill_between(x, y_min, y_max, color='gray', alpha=0.5)
    # Set the x-axis tick labels to res_df.exper, rotated for readability
    plt.xticks(ticks=x, labels=res_df.exper, rotation=45, fontsize =12)
    # Set labels for axes
    plt.xlabel('Experiments', fontsize=15, labelpad=10)
    plt.ylabel('Recall Rate', fontsize=15, labelpad=10)
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
    if y_lims:
        plt.ylim(y_lims)
    # Optionally, you can add a title
    plt.title(title, fontsize=15)
    plt.legend(fontsize = 15, loc='lower right')
    if title:
        plt.savefig(f"{title.replace(' ', '_')}.png", dpi=350, transparent=True, bbox_inches='tight')
    plt.show()
# %%
def df_by_series(serdf):
    df1 = serdf[serdf.exper.isin(['SchNet0', 'coAlignn1', 'coSchnet2', 'coAlignn3'])]
    df2 = serdf[~serdf.exper.isin(['SchNet0', 'coAlignn1', 'coSchnet2', 'coAlignn3'])]
    return df1, df2
def recall_y_margin(mardf):
    global_y_min = min(mardf.true_positive_rate.min(), mardf.LO_true_positive_rate.min())
    global_y_max = max(mardf.true_positive_rate.max(), mardf.LO_true_positive_rate.max())
    plot_margin = (global_y_max-global_y_min)*0.08
    y_lims = (global_y_min - plot_margin, global_y_max + plot_margin)
    return y_lims
    
# %%
plot_recall_iter(df_by_series(df)[0], y_lims=recall_y_margin(df), title= "SchNet0 Series Synthesizability Recall")
plot_recall_iter(df_by_series(df)[1], y_lims=recall_y_margin(df), title= "ALIGNN0 Series Synthesizability Recall")
# %%
plot_recall_iter(df_by_series(df015)[0], y_lims=recall_y_margin(df015), title= "SchNet0 Series Stability Recall")
plot_recall_iter(df_by_series(df015)[1], y_lims=recall_y_margin(df015), title= "ALIGNN0 Series Stability Recall")
# %%
# %%
# %%
plt.figure(figsize=(10, 6))  # Adjust the figure size as needed

# Plot y and y2 as lines
plt.plot(x, y, 'bo', label="TPR")  
plt.plot(x, y2, 'ro', label="LOTPR")  

# Calculate the minimum and maximum between y and y2 at each point
y_min = np.minimum(y, y2)
y_max = np.maximum(y, y2)

# Fill the area between y_min and y_max
plt.fill_between(x, y_min, y_max, color='gray', alpha=0.5)

# Plot y3
plt.plot(x, y3, color='green', marker='o', linestyle='-', label="GTTPR")  

# The rest of your code...

# The rest of your code...
# %%
# %%
# Assuming df is your DataFrame
x = range(len(df.exper))  # Create a range of numbers
y = df.true_positive_rate  # Values for y-axis
y2 = df.LO_true_positive_rate  # Create a range of numbers
# y3 = df.GT_true_positive_rate  # Create a range of numbers

plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
plt.plot(x, y, marker='o',label = "TPR")  # Plot with markers for each point
plt.plot(x, y2, marker='o',label = "LOTPR")  # Plot with markers for each point
# plt.plot(x, y3, marker='o',label = "GTTPR")  # Plot with markers for each point

# Set the x-axis tick labels to df.exper, rotated for readability
plt.xticks(ticks=x, labels=df.exper, rotation=45, fontsize =12)
# Set labels for axes
plt.xlabel('Experiment', fontsize=15, labelpad=10)
# plt.xlabel('Experiment')
plt.ylabel('True Positive Rate', fontsize=15, labelpad=10)
plt.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
# Optionally, you can add a title
plt.title('True Positive Rate Over Iterations', fontsize=15)
plt.legend(fontsize = 15)
# plt.savefig("tpr_iterations.png", dpi=350, transparent=True, bbox_inches='tight')
plt.show()
# %%
# Assuming df is your DataFrame
x = range(len(df.exper))  # Create a range of numbers
y = df.true_positive_rate  # Values for y-axis
y2 = df.LO_true_positive_rate  # Create a range of numbers
y3 = (y + y2) / 2 # Create a range of numbers

plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
# plt.plot(x, y, marker='o',label = "TPR")  # Plot with markers for each point
# plt.plot(x, y2, marker='o',label = "LOTPR")  # Plot with markers for each point
plt.plot(x, y3, marker='o',label = "Average True-Positive Rate")  # Plot with markers for each point

# # Set the x-axis tick labels to df.exper, rotated for readability
plt.xticks(ticks=x, labels=df.exper, rotation=45, fontsize =12)
# Set labels for axes
plt.xlabel('Experiment', fontsize=15, labelpad=10)
# plt.xlabel('Experiment')
plt.ylabel('True Positive Rate', fontsize=15, labelpad=10)
plt.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
# Optionally, you can add a title
plt.title('True Positive Rate Over Iterations', fontsize=15)
plt.legend(fontsize = 15)
plt.savefig("tpr_iterations_avg.png", dpi=350, transparent=True, bbox_inches='tight')
plt.show()
# %%
# %%
# Define the specific experiments to separate
alignn_exper = ['alignn0', 'coAl1', 'coAl2', 'coAl3']

# Subset of DataFrame for specific experiments
df_alignn = df[df.exper.isin(alignn_exper)]
x_alignn = range(len(df_alignn.exper))

# Subset of DataFrame for the rest
df_schnet = df[~df.exper.isin(alignn_exper)]
x_schnet = range(len(df_schnet.exper))

# Plot for specific experiments
plt.figure(figsize=(10, 6))
plt.plot(x_alignn, df_alignn.true_positive_rate, marker='o', label="TPR")
plt.plot(x_alignn, df_alignn.LO_true_positive_rate, marker='o', label="LOTPR")
# plt.plot(x_alignn, df_alignn.GT_true_positive_rate, marker='o', label="GTTPR")
plt.xticks(ticks=x_alignn, labels=df_alignn.exper, rotation=45)
plt.xlabel('Experiment')
plt.ylabel('True Positive Rate')
plt.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
plt.title('True Positive Rate for ALIGNN Iterations')
plt.legend()
plt.savefig("tpr_alignn_iterations.png", dpi=350, transparent=True, bbox_inches='tight')

plt.show()

# Plot for the rest of the experiments
plt.figure(figsize=(10, 6))
plt.plot(x_schnet, df_schnet.true_positive_rate, marker='o', label="TPR")
plt.plot(x_schnet, df_schnet.LO_true_positive_rate, marker='o', label="LOTPR")
# plt.plot(x_schnet, df_schnet.GT_true_positive_rate, marker='o', label="GTTPR")
plt.xticks(ticks=x_schnet, labels=df_schnet.exper, rotation=45)
plt.xlabel('Experiment')
plt.ylabel('True Positive Rate')
plt.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
plt.title('True Positive Rate for SCHNET Iterations')
plt.legend()
plt.savefig("tpr_schnet_iterations.png", dpi=350, transparent=True, bbox_inches='tight')
plt.show()

# %%
# Prepare data
x = range(len(df.exper))
y = df.true_positive_rate
y2 = df.LO_true_positive_rate
y3 = df.GT_true_positive_rate
y4 = df.predicted_positive_rate
y5 = df.false_positive_rate

# Create the plot
plt.figure(figsize=(12, 7))  # Adjust for optimal size
plt.plot(x, y, marker='o', color='blue', linestyle='-', label="TPR")
plt.plot(x, y2, marker='v', color='green', linestyle='-', label="LOTPR")
plt.plot(x, y3, marker='s', color='red', linestyle='-', label="GTTPR")
plt.plot(x, y4, marker='s', color='m', linestyle='-', label="PPR")
# plt.plot(x, y5, marker='s', color='k', linestyle='-', label="FPR")

# Enhance readability
plt.xticks(ticks=x, labels=df.exper, rotation=45, fontsize=12)
plt.xlabel('Experiment', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('True Positive Rate Over Experiments', fontsize=16)
plt.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
plt.legend(fontsize=12)

plt.show()

# %%
x = range(len(df.exper))
y = df.true_positive_rate
y2 = df.LO_true_positive_rate
y3 = df.GT_true_positive_rate
y4 = df.predicted_positive_rate

# Determine the range to exclude
# You might need to adjust these values based on your specific data
lower_range = 0.41
upper_range = .8
excluded_range = (lower_range, upper_range)

# Create the plot with broken axes
fig = plt.figure(figsize=(15, 10))  # Create a figure with a larger size

bax = brokenaxes(ylims=((0, lower_range), 
    (upper_range, max(y.max(), y2.max(), y3.max(), y4.max()) + 0.05)), hspace=.05, fig=fig)


# Plot the data
bax.plot(x, y, marker='o', color='blue', linestyle='-', label="TPR")
bax.plot(x, y2, marker='v', color='green', linestyle='-', label="LOTPR")
bax.plot(x, y3, marker='s', color='red', linestyle='-', label="GTTPR")
bax.plot(x, y4, marker='^', color='magenta', linestyle='-', label="PPR")

# Enhance readability and adjust axis titles
ticks = list(range(len(df.exper)))
labels = list(df.exper)
for ax in bax.axs:
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=45, fontsize=20)
# bax.set_xticklabels(labels, rotation=45, fontsize=20, ha = "center")
for ax in bax.axs:
    ax.set_yticklabels([round(i, 3) for i in ax.get_yticks()], fontsize=20)
bax.set_xlabel('Experiment', fontsize=26, labelpad=80)
bax.set_ylabel('True Positive Rate', fontsize=26, labelpad=60)
bax.set_title('True Positive Rate Over Experiments', fontsize=28)
bax.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
bax.legend(fontsize=20, loc='upper left', bbox_to_anchor=(.8, .3))

bax.axs[0].spines['top'].set_visible(True)
bax.axs[1].spines['right'].set_visible(True)

plt.show()

# %%
