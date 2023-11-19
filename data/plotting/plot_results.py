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
df = pd.read_csv("../results/stability015/results_plot.csv")
df = df.dropna()
# %%
# %%
# Assuming df is your DataFrame
x = range(len(df.exper))  # Create a range of numbers
y = df.true_positive_rate  # Values for y-axis
y2 = df.LO_true_positive_rate  # Create a range of numbers
y3 = df.GT_true_positive_rate  # Create a range of numbers

plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
plt.plot(x, y, marker='o',label = "TPR")  # Plot with markers for each point
plt.plot(x, y2, marker='o',label = "LOTPR")  # Plot with markers for each point
plt.plot(x, y3, marker='o',label = "GTTPR")  # Plot with markers for each point

# Set the x-axis tick labels to df.exper, rotated for readability
plt.xticks(ticks=x, labels=df.exper, rotation=45)
# Set labels for axes
plt.xlabel('Experiment')
plt.ylabel('True Positive Rate')
plt.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
# Optionally, you can add a title
plt.title('True Positive Rate Over Experiments')
plt.legend()

plt.show()

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
plt.plot(x_alignn, df_alignn.GT_true_positive_rate, marker='o', label="GTTPR")
plt.xticks(ticks=x_alignn, labels=df_alignn.exper, rotation=45)
plt.xlabel('Experiment')
plt.ylabel('True Positive Rate')
plt.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
plt.title('True Positive Rate for ALIGNN Experiments')
plt.legend()
plt.show()

# Plot for the rest of the experiments
plt.figure(figsize=(10, 6))
plt.plot(x_schnet, df_schnet.true_positive_rate, marker='o', label="TPR")
plt.plot(x_schnet, df_schnet.LO_true_positive_rate, marker='o', label="LOTPR")
plt.plot(x_schnet, df_schnet.GT_true_positive_rate, marker='o', label="GTTPR")
plt.xticks(ticks=x_schnet, labels=df_schnet.exper, rotation=45)
plt.xlabel('Experiment')
plt.ylabel('True Positive Rate')
plt.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
plt.title('True Positive Rate for SCHNET Experiments')
plt.legend()
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
