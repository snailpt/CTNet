import matplotlib.pyplot as plt
import numpy as np
import pandas as  pd
df_2a = pd.read_excel('heads.xlsx', '2a')
df_2b = pd.read_excel('heads.xlsx', '2b')
# Parameters and their corresponding accuracies for 9 subjects
# Updated parameters for both experiments
labels = [1, 2, 4, 8, 16]
labels = [str(i) for i in labels]
parameters_updated = np.array([1, 2, 3, 4, 5])
accuracies_2a = df_2a.T.values
accuracies_2b = df_2b.T.values
# Plotting both experiments in a subplot
fig, axs = plt.subplots(2, 1, figsize=(8, 8))

# Common settings for both subplots
colors = plt.cm.viridis(np.linspace(0, 1, 9))  # From warm to cold colors

colors = [
    "#1f77b4",  # Muted blue
    "#ff7f0e",  # Safety orange
    "#2ca02c",  # Cooked asparagus green
    "#d62728",  # Brick red
    "#9467bd",  # Muted purple
    "#8c564b",  # Chestnut brown
    "#e377c2",  # Raspberry yogurt pink
    "#7f7f7f",  # Middle gray
    "#bcbd22",  # Curry yellow-green
]
line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (3, 5, 1, 5)), (0, (5, 10)), 'dashdot', (0, (3, 10, 1, 10))]
marker_styles = ['o',  # Circle
                 's',  # Square
                 '^',  # Upward triangle
                 'D',  # Diamond
                 'p',  # Pentagon
                 '*',  # Star
                 'H',  # Hexagon
                 'x',  # X (cross)
                 '+']  # Plus


# BCI IV 2a Experiment
for i in range(9):
    axs[0].plot(parameters_updated, accuracies_2a[:, i], label=f'A0{i+1}', 
            color=colors[i], linestyle='-', marker=marker_styles[i], markersize=5)
    # axs[0].plot(parameters_updated, accuracies_2a[:, i], label=f'Subject {i+1}', color=colors[i], linestyle=line_styles[i])
axs[0].set_title('BCI IV-2a',fontsize=15)
axs[0].set_xlabel('Heads',fontsize=15)

axs[0].set_ylabel('Accuracy (%)',fontsize=15)
axs[0].set_ylim(50, 100)

axs[0].set_xticks(parameters_updated)
axs[0].set_xticklabels(labels,fontsize=13)
axs[0].tick_params(axis='y', labelsize=13)  # Adjusting y-axis tick font size

axs[0].set_ylim(55,100)
axs[0].legend()

# BCI IV 2b Experiment
for i in range(9):
    axs[1].plot(parameters_updated, accuracies_2b[:, i], label=f'B0{i+1}', 
            color=colors[i], linestyle='-', marker=marker_styles[i], markersize=5)
axs[1].set_title('BCI IV-2b',fontsize=15)
axs[1].set_xlabel('Heads',fontsize=15)
axs[1].set_ylabel('Accuracy (%)',fontsize=15)
axs[1].set_xticks(parameters_updated)
axs[1].set_xticklabels(labels,fontsize=13)
# axs[1].legend()
axs[1].set_ylim(55,100)
axs[1].tick_params(axis='y', labelsize=13)  # Adjusting y-axis tick font size


# Adjusting legends outside the plotting area on the right side
fig.subplots_adjust(right=0.8)
axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.69), shadow=True)
axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.69), shadow=True)

plt.tight_layout()
plt.tight_layout()
plt.savefig('Fig7.pdf', dpi=300, bbox_inches='tight')  

plt.show()
