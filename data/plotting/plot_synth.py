# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import os
from matplotlib.colors import LogNorm
# import seaborn as sns
# %%
prop = 'synth'
df = pd.read_pickle("../clean_data/synthDF")
df = df[df.energy_above_hull<15]
codf = pd.read_pickle("/home/samariam/projects/SynthCoTrain/data/results/synth/coSchAl2.pkl")
codf = codf.dropna()
# %%
# df.columns
# %%
# %%
duplicated_columns = df.columns[df.columns.isin(codf.columns) & (df.columns != 'material_id')]
codf_ = codf.drop(columns=duplicated_columns)
mdf = pd.merge(df, codf_, on='material_id')
mdf = mdf[mdf.energy_above_hull<10]
tdf_ = mdf[mdf[prop]==0]
edf_ = mdf[mdf[prop]==1]
# %%
mdf.predScore.hist()
# %%
plt.scatter(mdf.predScore, mdf.energy_above_hull)
plt.title("mdf")
# %%
plt.scatter(tdf_.predScore, tdf_.energy_above_hull)
plt.title("tdf")

# %%
# plt.scatter(edf_.predScore, edf_.energy_above_hull)
# plt.title("edf")
# %%
def save_plot(figure, filename):
    """
    Save a matplotlib figure to a file.

    Parameters:
        figure (matplotlib.figure.Figure): The figure to save.
        filename (str): The name of the file to save the figure as.
    """
    figure.savefig(filename)
# %%
def heatmap(codf, datadf, filename = None):
    plot_df = codf.merge(datadf[["material_id", "formation_energy_per_atom", "energy_above_hull"]], on="material_id")
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
    plt.hist2d(x, y, bins=(50, 50), cmap='viridis', norm=LogNorm(vmin=vmin, vmax=vmax), cmin=1)
    cbar = plt.colorbar()
    plt.xlabel('predScore')
    plt.ylabel('energy_above_hull')
    plt.title('Density Heatmap')
    if filename:
        save_plot(plt.gcf(), filename)
    plt.show()
# %%
def heatmapZoom(codf, datadf, ehull_cutoff = 1, filename=None):
    plot_df = codf.merge(datadf[["material_id", "formation_energy_per_atom", "energy_above_hull"]], on="material_id")
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


    plt.hist2d(xzoomed, yzoomed, bins=(50, 50), cmap='viridis', norm=LogNorm(vmin=vmin, vmax=vmax), cmin=1)
    cbar = plt.colorbar()
    plt.xlabel('predScore')
    plt.ylabel('energy_above_hull')
    plt.title('Zoomed in Density Heatmap')
    if filename:
        save_plot(plt.gcf(), filename)
    plt.show()

# %%
# Assuming tdf.predScore and tdf.energy_above_hull are your data columns
def density_colors(x,y):
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
    
#%% 
def scatter_hm(codf, datadf, filename=None):
    plot_df = codf.merge(datadf[["material_id", "formation_energy_per_atom", "energy_above_hull"]], on="material_id")
    colors, x, y = density_colors(plot_df.predScore, plot_df.energy_above_hull)
    # Plot first scatter plot
    plt.scatter(x, y, c=colors, cmap='viridis', norm=LogNorm(), alpha=0.7, s=20)
    plt.colorbar()
    plt.xlabel('predScore')
    plt.ylabel('energy_above_hull')
    plt.title('Density Scatter Plot')
    if filename:
        save_plot(plt.gcf(), filename)
    plt.show()
# %%
def scatter_hm_zoomed(codf,datadf, ehull_cutoff = 1, filename=None):
    plot_df = codf.merge(datadf[["material_id", "formation_energy_per_atom", "energy_above_hull"]], on="material_id")
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
    x_indices_zoomed = np.clip(np.digitize(x_zoomed, xedges) - 1, 0, heatmap.shape[0] - 1)
    y_indices_zoomed = np.clip(np.digitize(y_zoomed, yedges) - 1, 0, heatmap.shape[1] - 1)
    colors_zoomed = heatmap[x_indices_zoomed, y_indices_zoomed]

    # Plot second (zoomed-in) scatter plot
    plt.scatter(x_zoomed, y_zoomed, c=colors_zoomed, cmap='viridis', 
                norm=LogNorm(vmin=vmin, vmax=vmax), alpha=0.7, s=20)
    plt.colorbar()
    plt.xlabel('predScore (Zoomed)')
    plt.ylabel('energy_above_hull (Zoomed)')
    plt.title('Zoomed Density Scatter Plot')
    if filename:
        save_plot(plt.gcf(), filename)
    plt.show()
    
# %%
def label_dist(codf,datadf, ehull = False, prop = prop, filename=None):
    plot_df = codf.merge(datadf[["material_id", "formation_energy_per_atom", "energy_above_hull"]], on="material_id")
    if ehull:
        prop = "stability"
    prediction = "prediction"        
    edf = plot_df[plot_df[prop]==1]
    tdf = plot_df[plot_df[prop]==0]
    edf = edf.sort_values(prediction, ascending=False)
    tdf = tdf.sort_values(prediction, ascending=True)
    
    print(f"edf prediction means is {edf.prediction.mean()}. Length of edf is {len(edf)}.")
    print(f"tdf prediction means is {tdf.prediction.mean()}. Length of tdf is {len(tdf)}.")
    
    data_classes_e = ['False (-) class', 'True (+) class']
    data_classes_t = ['Predicted (-) class', 'Predicted (+) class']

    colors = ListedColormap(['r','b'])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6,7))

    scatter = ax1.scatter(
        edf.energy_above_hull, 
        edf.formation_energy_per_atom,
        c=edf['prediction'],  # Make sure 'prediction' is quoted if it's a column name
        cmap=colors, alpha=.6
    )

    ax1.legend(handles=scatter.legend_elements()[0], labels=data_classes_e)
    ax1.set_title('Prediction Distribution of Experimental Data', fontsize=13.5, fontweight='bold')

    # Use ax2 directly to create the scatter plot instead of calling plt.subplot again
    scatter2 = ax2.scatter(
        tdf.energy_above_hull, 
        tdf.formation_energy_per_atom,
        c=tdf['prediction'],  # Again, quote 'prediction' if it's a column name
        cmap=colors, alpha=.5
    )

    # If you want to modify tick parameters for ax1, do it with ax1.tick_params
    ax1.tick_params('x', labelbottom=False)

    # Set the title for ax2 as well, and any other configurations you need
    ax2.set_title('Prediction Distribution of Theoretical Data', fontsize=13.5, fontweight='bold')

    ax2.legend(handles=scatter.legend_elements()[0], labels=data_classes_t)

    ax2.set_xlabel('Energy above hull', fontsize=15)
    # ax2.set_ylabel('Formation energy per atom', fontsize=12)
    ax2.set_title('Prediction Distribution of Theorertical Data', fontsize=13.5, fontweight='bold');

    plt.subplots_adjust(hspace=.15)
    fig.text(0.04, 0.5, 'Formation energy per atom', ha='center', va='center',
            rotation='vertical', fontsize=16.5);
    if filename:
        save_plot(plt.gca(), filename)

# %%

def label_dist2(codf, datadf, prop = prop, ehull=False, filename=None):
    plot_df = codf.merge(datadf[["material_id", "formation_energy_per_atom", "energy_above_hull"]], on="material_id")
    if ehull:
        prop = "stability"
    edf = plot_df[plot_df[prop] == 1]
    tdf = plot_df[plot_df[prop] == 0]
    edf = edf.sort_values('prediction', ascending=False)
    tdf = tdf.sort_values('prediction', ascending=True)

    colors = ListedColormap(['r', 'b'])
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))  # 2x2 subplot grid

    # Define the plots for experimental data
    scatter1 = axs[0, 0].scatter(
        edf.energy_above_hull[edf['prediction'] == 0],
        edf.formation_energy_per_atom[edf['prediction'] == 0],
        c=edf['prediction'][edf['prediction'] == 0],
        cmap=colors, alpha=.6
    )

    axs[0, 0].set_title('Experimental Data (Negative Label)', fontsize=13.5, fontweight='bold')

    scatter2 = axs[1, 0].scatter(
        edf.energy_above_hull[edf['prediction'] == 1],  # Filter for minority class if needed
        edf.formation_energy_per_atom[edf['prediction'] == 1],
        c='blue', alpha=.6
    )

    axs[1, 0].set_title('Experimental Data (Positive Label)', fontsize=13.5, fontweight='bold')

    # Define the plots for theoretical data
    scatter3 = axs[0, 1].scatter(
        tdf.energy_above_hull[tdf['prediction'] == 0],
        tdf.formation_energy_per_atom[tdf['prediction'] == 0],
        c=tdf['prediction'][tdf['prediction'] == 0],
        cmap=colors, alpha=.5
    )

    axs[0, 1].set_title('Theoretical Data (Negative Label)', fontsize=13.5, fontweight='bold')

    scatter4 = axs[1, 1].scatter(
        tdf.energy_above_hull[tdf['prediction'] == 1],
        tdf.formation_energy_per_atom[tdf['prediction'] == 1],
        c='b', alpha=.5
    )

    axs[1, 1].set_title('Theoretical Data (Positive Label)', fontsize=13.5, fontweight='bold')

    # Set common labels
    for ax in axs.flat:
        # ax.set(xlabel='Energy above hull', ylabel='Formation energy per atom')
        ax.label_outer()  # Only show outer labels

    # Adjust subplot params
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    # Add a big axis, hide frame
    big_ax = fig.add_subplot(111, frameon=False)
    # Hide tick and tick label of the big axis
    big_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    big_ax.set_ylabel('Formation energy per atom', labelpad=20, fontsize=16.5)
    big_ax.set_xlabel('Energy above hull', labelpad=20, fontsize=16.5)

    if filename:
        plt.savefig(filename)

    plt.show()
# %%
def label_dist3(codf, datadf, ehull=False,prop = prop, filename=None):
    plot_df = codf.merge(datadf[["material_id", "formation_energy_per_atom", "energy_above_hull"]], on="material_id")
    if ehull:
        prop = "stability"
    edf = plot_df[plot_df[prop] == 1]
    tdf = plot_df[plot_df[prop] == 0]
    edf = edf.sort_values('prediction', ascending=False)
    tdf = tdf.sort_values('prediction', ascending=True)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))  # 2x2 subplot grid
    # Define the plots for experimental data
    colors, x, y = density_colors(edf.energy_above_hull[edf['prediction'] == 0],
                                  edf.formation_energy_per_atom[edf['prediction'] == 0])
    scatter1 = axs[0, 0].scatter(x,y,c=colors, cmap='viridis', norm=LogNorm(), alpha=0.7, )
    fig.colorbar(scatter1, ax=axs[0, 0], orientation='vertical')
    axs[0, 0].set_title('Experimental Data (Negative Label)', fontsize=13.5, fontweight='bold')

    colors, x, y = density_colors(edf.energy_above_hull[edf['prediction'] == 1],  
        edf.formation_energy_per_atom[edf['prediction'] == 1])
    scatter2 = axs[1, 0].scatter(x,y,c=colors, cmap='viridis', norm=LogNorm(), alpha=0.7, )
    fig.colorbar(scatter2, ax=axs[1, 0], orientation='vertical')
    axs[1, 0].set_title('Experimental Data (Positive Label)', fontsize=13.5, fontweight='bold')

    # # Define the plots for theoretical data
    colors, x, y = density_colors(tdf.energy_above_hull[tdf['prediction'] == 0],
                                  tdf.formation_energy_per_atom[tdf['prediction'] == 0])
    scatter3 = axs[0, 1].scatter(x,y,c=colors, cmap='viridis', norm=LogNorm(), alpha=0.7, )
    fig.colorbar(scatter3, ax=axs[0, 1], orientation='vertical')
    axs[0, 1].set_title('Theoretical Data (Negative Label)', fontsize=13.5, fontweight='bold')

    colors, x, y = density_colors(tdf.energy_above_hull[tdf['prediction'] == 1],  
        tdf.formation_energy_per_atom[tdf['prediction'] == 1])
    scatter4 = axs[1, 1].scatter(x,y,c=colors, cmap='viridis', norm=LogNorm(), alpha=0.7, )
    fig.colorbar(scatter4, ax=axs[1, 1], orientation='vertical')
    axs[1, 1].set_title('Theoretical Data (Positive Label)', fontsize=13.5, fontweight='bold')

    # Set common labels
    for ax in axs.flat:
        # ax.set(xlabel='Energy above hull', ylabel='Formation energy per atom')
        ax.label_outer()  # Only show outer labels

    # Adjust subplot params
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    # Add a big axis, hide frame
    big_ax = fig.add_subplot(111, frameon=False)
    # Hide tick and tick label of the big axis
    big_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    big_ax.set_ylabel('Formation energy per atom', labelpad=20, fontsize=16.5)
    big_ax.set_xlabel('Energy above hull', labelpad=20, fontsize=16.5)

    if filename:
        plt.savefig(filename)

    plt.show()

# %%
# %%
# def midlabel_dist(codf, datadf, ehull=False,prop = prop, filename=None):
    # plot_df = codf.merge(datadf[["material_id", "formation_energy_per_atom", "energy_above_hull"]], on="material_id")
    # if ehull:
    #     prop = "stability"
    # edf = plot_df[plot_df[prop] == 1]
    # tdf = plot_df[plot_df[prop] == 0]
    # edf = edf.sort_values('prediction', ascending=False)
    # tdf = tdf.sort_values('prediction', ascending=True)
    # true_positive_rate = edf.prediction.mean()
    # unlabeled_synth_frac = tdf.prediction.mean()
    # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6,6))
    # ax1.hist(tdf.predScore, bins=40)
    # # edf.Preds.hist(bins=40)
    # ax1.set_title('Probability Distribution for Experimental Data', fontsize=13.5, fontweight='bold')
    # ax1.vlines(.5, 0,900, 'r',linewidth=3)
    # ax1.arrow(.5,600,.35,0, linewidth=3, color ='r', length_includes_head=True,
    #         head_width=75, head_length=0.05)
    # ax1.text(.55,350, '{:.1f}% true- \npositive rate'.format(true_positive_rate*100), fontsize = 15);


    # plt.xlabel('Predicted Probability of Synthesizability', fontsize=14)

    # ax2.hist(tdf.coSchAl2,bins=40)
    # ax2.vlines(.5, 0,24500, 'r',linewidth=3)
    # ax2.arrow(.5,16000,.35,0, linewidth=3, color ='r', length_includes_head=True,
    #         head_width=1800, head_length=0.05)
    # ax2.set_title('Probability Distribution for Theoretical Data', fontsize=13.5,fontweight='bold')
    # ax2.text(.55,9500, '{:.1f}% predicted \n synthesizable'.format(unlabeled_synth_frac*100), fontsize = 15);



# %%
