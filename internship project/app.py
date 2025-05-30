from flask import Flask, request, jsonify, render_template, send_from_directory
import pandas as pd
import os
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Helper function to process data and detect anomalies
def process_data(file_path):
    data = pd.read_csv(file_path)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data[['age', 'lab_test_result']])
    iso_forest = IsolationForest(contamination=0.2, random_state=42)
    data['anomaly'] = iso_forest.fit_predict(data_scaled)
    return data

# Helper function to train model
def train_model(data):
    X_train, X_test, y_train, y_test = train_test_split(
        data[['age', 'lab_test_result']], data['diagnosis'], test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(random_state=42, n_estimators=50)
    model.fit(X_train, y_train)
    return model, X_test, y_test

# Helper function to create and save pie chart
def create_pie_chart(anomaly_counts, filename="anomaly_pie_chart.png"):
    labels = ['Normal', 'Anomalous']
    sizes = [anomaly_counts.get(1, 0), anomaly_counts.get(-1, 0)]

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,
           colors=['#4CAF50', '#FF5722'])
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    filepath = os.path.join(STATIC_FOLDER, filename)
    plt.savefig(filepath, format='png')
    plt.close(fig)
    return filepath

# Route: home page
@app.route("/")
def index():
    return render_template("index.html")

# Route: handle file upload and analysis
@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Process and analyze the data
    data = process_data(file_path)
    anomaly_counts = data['anomaly'].value_counts().to_dict()

    # Generate pie chart and save to static
    create_pie_chart(anomaly_counts)

    # Train model and get accuracy
    model, X_test, y_test = train_model(data)
    model_score = model.score(X_test, y_test)

    return jsonify({
        "anomaly_counts": anomaly_counts,
        "model_score": model_score,
        "pie_chart_url": "/static/anomaly_pie_chart.png"
    })

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
