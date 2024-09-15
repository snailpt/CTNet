import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load data for BCI IV-2a
df_a = pd.read_excel('length.xlsx', sheet_name='2a')
# Load data for BCI IV-2b
df_b = pd.read_excel('length.xlsx', sheet_name='2b')

# Set token sizes that were experimented with
param_values = [12, 13, 15, 17, 20, 25, 31, 41, 62, 125]
param_labels = [str(i) for i in param_values]

# Extract mean values and standard deviations for both datasets
mean_values_a = df_a['accuracy']
std_devs_a = df_a['std']
mean_values_b = df_b['accuracy']
std_devs_b = df_b['std']

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

# Plot for BCI IV-2a
ax1.errorbar(param_labels, mean_values_a, yerr=std_devs_a, fmt='-o', ecolor='red', capsize=5, capthick=2, color='blue', label='BCI IV-2a')
ax1.set_title('BCI IV-2a', fontsize=14)
ax1.set_xlabel('Token size', fontsize=13)
ax1.set_ylabel('Accuracy (%)', fontsize=13)
ax1.set_xticks(param_labels)
ax1.set_ylim(65, 100)
ax1.grid(True, which='major', linestyle='--', linewidth='0.5', color='grey')
# ax1.legend()

# Plot for BCI IV-2b
ax2.errorbar(param_labels, mean_values_b, yerr=std_devs_b, fmt='-o', ecolor='red', capsize=5, capthick=2, color='blue', label='BCI IV-2b')
ax2.set_title('BCI IV-2b', fontsize=14)
ax2.set_xlabel('Token size', fontsize=13)
ax2.set_ylabel('Accuracy (%)', fontsize=13)
ax2.set_xticks(param_labels)
ax2.set_ylim(65, 100)
ax2.grid(True, which='major', linestyle='--', linewidth='0.5', color='grey')
# ax2.legend()

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig('Fig6.pdf', dpi=300, bbox_inches='tight')

# Display the plot
plt.show()
