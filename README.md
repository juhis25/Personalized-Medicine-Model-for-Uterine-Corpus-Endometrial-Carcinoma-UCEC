# UCEC Personalized Medicine Framework

This repository contains code for developing and deploying a personalized medicine framework for Uterine Corpus Endometrial Carcinoma (UCEC) using Python.

## Project Overview

- **Data Acquisition**: Obtaining and preprocessing genomic data from sources like TCGA.
- **Machine Learning**: Developing sophisticated machine learning models for predictive analysis.
- **Deployment**: Deploying the model using Flask for real-time predictions.
- **Version Control**: Using Git for robust version control.
- **Database Management**: Using SQL for enhanced database management.

## Installation

To run this project, you need to have Python installed. Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the necessary packages.

```bash
pip install -r requirements.txt
```

```markdown
# Detailed Explanation of Scripts

This document provides a detailed explanation of each script used in the UCEC Personalized Medicine Framework project.

```

Scripts

### 1. Data Acquisition

**File: `data_acquisition.py`**

This script is responsible for downloading genomic data from the TCGA database.

```python
import pandas as pd
import requests
import os

def download_tcga_data(project, data_type, output_dir):
    url = f"https://api.gdc.cancer.gov/data?filters={{%22op%22:%22and%22,%22content%22:[{{%22op%22:%22in%22,%22content%22:{{%22field%22:%22cases.project.project_id%22,%22value%22:[%22{project}%22]}}}},{{%22op%22:%22in%22,%22content%22:{{%22field%22:%22files.data_type%22,%22value%22:[%22{data_type}%22]}}}}]}}&pretty=true&size=1000&format=TSV"
    response = requests.get(url)
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/{project}_{data_type}.tsv", 'wb') as f:
        f.write(response.content)

project = "TCGA-UCEC"
data_type = "Gene Expression Quantification"
output_dir = "../data"
download_tcga_data(project, data_type, output_dir)
```

**Steps:**
1. Import necessary libraries.
2. Define a function `download_tcga_data` that constructs the API URL based on the project and data type.
3. Send a request to the TCGA API to fetch data.
4. Save the downloaded data into the specified output directory.

### 2. Data Preprocessing

**File: `data_preprocessing.py`**

This script normalizes the gene expression data.

```python
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
```

**Steps:**
1. Import necessary libraries.
2. Load the gene expression data from the downloaded file.
3. Normalize the data using `StandardScaler` from scikit-learn.
4. Save the normalized data to a new CSV file.

### 3. Feature Selection and Engineering

**File: `feature_selection.py`**

This script performs feature selection using ANOVA F-test.

```python
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
```

**Steps:**
1. Import necessary libraries.
2. Load the normalized gene expression data and the labels.
3. Perform ANOVA F-test to select significant features.
4. Save the selected features to a new CSV file.

### 4. Machine Learning Model Development

**File: `model_training.py`**

This script trains a RandomForestClassifier on the selected features.

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# Load data
selected_features_df = pd.read_csv('../data/selected_features.csv', index_col=0)
labels = pd.read_csv('../data/labels.csv')

# Split the data
X_train, X_test, y_train, y_test = train_test_split(selected_features_df, labels, test_size=0.2, random_state=42)

# Train the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train.values.ravel())

# Evaluate the model
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy}")

# Save the model
os.makedirs('../models', exist_ok=True)
joblib.dump(rf_model, '../models/rf_model.joblib')
```

**Steps:**
1. Import necessary libraries.
2. Load the selected features and labels.
3. Split the data into training and testing sets.
4. Train a RandomForestClassifier on the training data.
5. Evaluate the model on the testing data and print the accuracy.
6. Save the trained model using `joblib`.

### 5. Deployment

**File: `app.py`**

This script creates a Flask web application to serve the trained model for real-time predictions.

```python
from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('../models/rf_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame(data)
    predictions = model.predict(df)
    return jsonify({'predictions': predictions.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
```

**Steps:**
1. Import necessary libraries.
2. Create a Flask web application.
3. Load the trained model.
4. Define a route `/predict` that accepts POST requests with JSON data.
5. Convert the JSON data to a DataFrame and use the model to make predictions.
6. Return the predictions as a JSON response.
7. Run the Flask application.

## Additional Files

### .gitignore

This file specifies files and directories that should be ignored by Git.

```
# Ignore Python cache files
__pycache__/
*.pyc

# Ignore data files
data/

# Ignore model files
models/

# Ignore virtual environment
venv/
```

### requirements.txt

This file lists all the Python packages required to run the project.

```
pandas
scikit-learn
joblib
flask
requests
```


