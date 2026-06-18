🛡️ PassShield IDS — Backend API
📌 Overview

PassShield IDS Backend is a machine learning-powered Intrusion Detection System (IDS) API built with Flask. It analyzes network traffic data (NSL-KDD format CSV/TXT files) and classifies them into Normal or various attack types using a trained Random Forest model.

The backend is responsible for:

Processing uploaded network traffic datasets
Preprocessing and encoding categorical + numerical features
Running ML-based intrusion detection
Returning structured analytics for frontend visualization
Generating risk scores and attack distribution metrics
⚙️ Tech Stack
Python 3.10+
Flask
Flask-CORS
scikit-learn
pandas
NumPy
joblib
Gunicorn
🧠 Machine Learning Model

The backend uses a Random Forest Classifier trained on NSL-KDD dataset features.

Model Inputs:
Network traffic features (41–43 columns depending on dataset version)
Categorical features:
protocol_type
service
flag
Numerical features:
src_bytes
dst_bytes
duration
etc.
Output Classes:
Normal
DoS (Denial of Service)
Probe
R2L (Remote to Local)
U2R (User to Root)
📁 Project Structure
backend/
│
├── app.py # Main Flask API
├── requirements.txt # Python dependencies
│
├── models/
│ ├── intrusion_model.pkl # Trained ML model
│ ├── label_encoders.pkl # Encoders + scaler + metadata
│ ├── metrics.json # Baseline performance metrics
│
├── utils/
│ ├── attack_mapping.py # Maps raw labels to attack categories
│ ├── preprocessing.py # Custom preprocessing utilities
🚀 API Endpoints

1. Health Check
   GET /health

Checks if the API and ML model are loaded.

Response:
{
"status": "healthy",
"model_loaded": true,
"threshold": 0.35
} 2. Intrusion Detection (Main Endpoint)
POST /analyze

Analyzes uploaded network traffic dataset.

Request:
Form-data file upload
Accepts:
.csv
.txt (NSL-KDD format)
Example (Frontend):
const formData = new FormData();
formData.append("file", file);

fetch("https://your-api.onrender.com/analyze", {
method: "POST",
body: formData
});
What it does internally:
Loads dataset using pandas
Detects dataset format (41–43 columns)
Maps attack labels (if present)
Encodes categorical features
Scales numerical features
Runs ML prediction using Random Forest
Applies probability threshold filtering
Classifies traffic into:
Normal
Attack categories
Response Structure:
{
"summaryStats": {
"totalRecords": 1000,
"normalTraffic": 700,
"threatsDetected": 300,
"riskLevel": "High"
},
"attackDistribution": {
"pie": {
"labels": ["Normal", "DoS", "Probe", "R2L", "U2R"],
"data": [70, 15, 10, 4, 1]
},
"bar": {
"labels": ["DoS", "Probe", "R2L", "U2R"],
"data": [150, 100, 40, 10]
}
},
"performanceMetrics": {
"accuracy": 98.45,
"precision": 97.80,
"recall": 96.90,
"f1Score": 97.35,
"falsePositiveRate": 0.85
},
"predictions": [
{
"id": "REC-0001",
"timestamp": "2026-06-18 10:00:00",
"protocol": "TCP",
"srcIp": "192.168.1.12",
"dstIp": "10.0.0.5",
"packetSize": 1024,
"prediction": "Normal"
}
]
}
📊 Features
✔ Machine Learning IDS

Detects and classifies network intrusions using trained model.

✔ Risk Scoring Engine

Automatically assigns:

Low
Medium
High
Critical

based on attack percentage.

✔ Dataset Compatibility

Supports:

NSL-KDD CSV format
KDD TXT format
✔ Intelligent Preprocessing
Label encoding for categorical features
Feature scaling for numerical values
Handles missing/unknown columns safely
✔ Attack Mapping

Converts raw attack labels into standardized categories:

DoS
Probe
R2L
U2R
🌐 Deployment
Production Hosting:

Render

Start Command:
gunicorn app:app
Build Command:
pip install -r requirements.txt
⚠️ Known Constraints
Large datasets may increase processing time
Model must be pre-trained before deployment
Requires correct NSL-KDD formatted input
Runs synchronously (no streaming yet)
🔐 Security Notes
CORS enabled for frontend integration
No authentication layer implemented yet (future upgrade)
File validation enforced for CSV/TXT only
📌 Future Improvements
Real-time packet streaming (WebSockets)
Live network monitoring mode
API authentication (JWT)
Database logging of attacks
Dockerized deployment
Model retraining pipeline
👨‍💻 Author

PassShield IDS Backend
Built as part of a Cybersecurity + Machine Learning intrusion detection system.
