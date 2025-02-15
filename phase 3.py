import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.tokenize import word_tokenize

# Sample structured data
# This dataset contains three columns: 'age', 'lab_test_result', and 'diagnosis'.
# 'age' represents the age of the patients, 'lab_test_result' contains results from a lab test, and 'diagnosis' is a binary label (e.g., 0 = no disease, 1 = disease).
data = pd.DataFrame({
    'age': [25, 30, 45, 40, 50],
    'lab_test_result': [5.1, 4.9, 6.0, 5.8, 7.1],
    'diagnosis': [0, 1, 0, 1, 0]
})

# ======================
# Data Visualization
# ======================

# Histogram for age distribution
# This plot visualizes the distribution of the 'age' column to identify patterns or anomalies in patient demographics.
sns.histplot(data['age'], kde=True, bins=5)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# ======================
# Anomaly Detection
# ======================

# Feature scaling
# Before applying anomaly detection, features ('age' and 'lab_test_result') are standardized using StandardScaler.
# This ensures that all features have a mean of 0 and a standard deviation of 1, making the model more robust to varying scales.
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[['age', 'lab_test_result']])

# Isolation Forest
# Anomaly detection is performed using Isolation Forest. This algorithm is effective for high-dimensional data and detects anomalies by isolating them in a decision tree structure.
# The 'contamination' parameter specifies the proportion of anomalies expected in the data (20% in this case).
iso_forest = IsolationForest(contamination=0.2, random_state=42)
data['anomaly'] = iso_forest.fit_predict(data_scaled)

# Map anomaly scores to readable labels
# Anomalies (-1) and normal data points (1) are mapped to human-readable labels for easier interpretation.
data['anomaly_label'] = data['anomaly'].map({1: 'Normal', -1: 'Anomaly'})
print("Anomaly Detection Results:")
print(data[['age', 'lab_test_result', 'anomaly_label']])

# ======================
# Basic NLP
# ======================

# Tokenizing a clinical note
# Tokenization is a crucial step in natural language processing (NLP). Here, the 'punkt' tokenizer from NLTK splits the input text into individual words or tokens.
nltk.download('punkt', quiet=True)
clinical_note = "Patient shows elevated blood pressure."
words = word_tokenize(clinical_note)
print("Tokenized Words:", words)