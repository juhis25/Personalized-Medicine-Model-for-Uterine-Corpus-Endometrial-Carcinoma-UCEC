import pandas as pd
from sklearn.feature_selection import f_classif
import os

# Load data
normalized_df = pd.read_csv('../data/normalized_gene_expression.csv', index_col=0)
labels = pd.read_csv('../data/labels.csv')

# Perform ANOVA F-test
f_values, p_values = f_classif(normalized_df, labels.values.ravel())
selected_features = normalized_df.columns[p_values < 0.05]

# Save selected features
selected_features_df = normalized_df[selected_features]
os.makedirs('../data', exist_ok=True)
selected_features_df.to_csv('../data/selected_features.csv', index=True)
