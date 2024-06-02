import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
data_file_path = '/mnt/data/cancer-distribution-table.2024-05-30.tsv'
data = pd.read_csv(data_file_path, sep='\t')

# Extract relevant columns and preprocess data
data_clean = data[['Project', '# SSM Affected Cases', '# CNV Gains', '# CNV Losses', '# Mutations']]

# Convert percentage values to numeric
data_clean['# SSM Affected Cases'] = data_clean['# SSM Affected Cases'].str.extract(r'(\d+\.\d+)').astype(float)
data_clean['# CNV Gains'] = data_clean['# CNV Gains'].str.extract(r'(\d+\.\d+)').astype(float)
data_clean['# CNV Losses'] = data_clean['# CNV Losses'].str.extract(r'(\d+\.\d+)').astype(float)

# Set 'Project' as index
data_clean.set_index('Project', inplace=True)

# Plot heatmap with customizations
plt.figure(figsize=(14, 10))
heatmap = sns.heatmap(
    data_clean, 
    annot=True, 
    fmt=".2f", 
    cmap='coolwarm',  # Custom color map
    linewidths=.5,  # Line width for grid lines
    linecolor='gray',  # Color of grid lines
    cbar_kws={'label': 'Value'},  # Color bar label
    annot_kws={"size": 10}  # Annotation font size
)

# Customize axes and title
plt.title('Customized Cancer Data Heatmap', fontsize=18, pad=20)
plt.xlabel('Metrics', fontsize=14)
plt.ylabel('Cancer Projects', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(rotation=0, fontsize=12)

# Save and show the plot
custom_heatmap_path = '/mnt/data/custom_cancer_data_heatmap.png'
plt.savefig(custom_heatmap_path, bbox_inches='tight')
plt.show()
