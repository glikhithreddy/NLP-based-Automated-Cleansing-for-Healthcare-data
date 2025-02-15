import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.tokenize import word_tokenize

# Sample structured data
data = pd.DataFrame({
    'age': [25, 30, 45, 40, 50],
    'lab_test_result': [5.1, 4.9, 6.0, 5.8, 7.1],
    'diagnosis': [0, 1, 0, 1, 0]
})

# ======================
# Data Visualization
# ======================

# Histogram for age distribution
sns.histplot(data['age'], kde=True, bins=5)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# ======================
# Anomaly Detection
# ======================
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[['age', 'lab_test_result']])
iso_forest = IsolationForest(contamination=0.2, random_state=42)
data['anomaly'] = iso_forest.fit_predict(data_scaled)

# Map anomaly scores to readable labels
data['anomaly_label'] = data['anomaly'].map({1: 'Normal', -1: 'Anomaly'})
print("Anomaly Detection Results:")
print(data[['age', 'lab_test_result', 'anomaly_label']])

# ======================
# Basic NLP
# ======================
nltk.download('punkt', quiet=True)
clinical_note = "Patient shows elevated blood pressure."
words = word_tokenize(clinical_note)
print("Tokenized Words:", words)



