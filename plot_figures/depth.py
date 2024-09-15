# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 16:55:11 2024

@author: Administrator
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Adjusting the number of parameters to 8 and adding a second experiment scheme
param_values = np.arange(1, 11)  # Parameter values from 1 to 8
num_subjects = 9

# Generating random accuracy data for two experiment schemes
# Scheme 1
df = pd.read_excel('depth.xlsx', sheet_name='2a')
# param_values = np.arange(1, 9)  # Parameter values from 1 to 10
accuracy_data_scheme1 = df.T.values

# Scheme 2
df = pd.read_excel('depth.xlsx', sheet_name='2b')
accuracy_data_scheme2 = df.T.values

# Plotting violin plots with subplots for two experiment schemes
fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=False)

# Scheme 1 Violin Plot
data_for_plot_scheme1 = [accuracy_data_scheme1[i, :] for i in range(len(param_values))]
axs[0].violinplot(data_for_plot_scheme1, param_values, showmeans=False, showmedians=False)
# Adding mean values as solid dots and medians as lines for Scheme 1
for i in range(len(param_values)):
    pc_mean = np.mean(accuracy_data_scheme1[i, :])
    # pc_median = np.median(accuracy_data_scheme1[i, :])
    axs[0].plot(param_values[i], pc_mean, 'o', color='green', markersize=8)
    # axs[0].hlines(pc_median, param_values[i] - 0.1, param_values[i] + 0.1, color='black', lw=2)
axs[0].set_title('BCI IV-2a',fontsize=15)
axs[0].set_xticks(param_values)
axs[0].tick_params(axis='x', labelsize=13)  # Increase x-tick font size
axs[0].tick_params(axis='y', labelsize=13) 
axs[0].set_ylabel('Accuracy (%)',fontsize=15)
axs[0].grid(True, which='major', linestyle='--', linewidth='0.5', axis='y')  # Horizontal dashed grid lines
axs[0].set_ylim(60, 100)
axs[0].set_xlabel('Depth',fontsize=15)
# Scheme 2 Violin Plot
data_for_plot_scheme2 = [accuracy_data_scheme2[i, :] for i in range(len(param_values))]
axs[1].violinplot(data_for_plot_scheme2, param_values, showmeans=False, showmedians=False)
# Adding mean values as solid dots and medians as lines for Scheme 2
for i in range(len(param_values)):
    pc_mean = np.mean(accuracy_data_scheme2[i, :])
    # pc_median = np.median(accuracy_data_scheme2[i, :])
    axs[1].plot(param_values[i], pc_mean, 'o', color='green', markersize=8)
    # axs[1].hlines(pc_median, param_values[i] - 0.1, param_values[i] + 0.1, color='black', lw=2)
axs[1].set_title('BCI IV-2b',fontsize=15)
axs[1].tick_params(axis='x', labelsize=13)  # Increase x-tick font size
axs[1].tick_params(axis='y', labelsize=13) 
axs[1].set_xlabel('Depth',fontsize=15)
axs[1].set_ylabel('Accuracy (%)',fontsize=15)
axs[1].grid(True, which='major', linestyle='--', linewidth='0.5', axis='y')  # Horizontal dashed grid lines
axs[1].set_ylim(60, 100)
axs[1].set_xticks(param_values)
# plt.xticks(param_values)
plt.tight_layout()
plt.savefig('Fig8.pdf', dpi=300, bbox_inches='tight')  

plt.show()
