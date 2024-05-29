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
