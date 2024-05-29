import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# Load data
gene_expr_data = pd.read_csv('../data/TCGA-UCEC_Gene_Expression_Quantification.tsv', sep='\t')

# Normalize the data
scaler = StandardScaler()
normalized_data = scaler.fit_transform(gene_expr_data.iloc[:, 1:])
normalized_df = pd.DataFrame(normalized_data, columns=gene_expr_data.columns[1:], index=gene_expr_data.iloc[:, 0])

# Save normalized data
os.makedirs('../data', exist_ok=True)
normalized_df.to_csv('../data/normalized_gene_expression.csv', index=True)
