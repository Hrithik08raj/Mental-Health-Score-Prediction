# 🧠 MindScore – Mental Health Score Prediction

A complete full-stack machine learning web application that predicts a student's wellbeing score (0 to 10) based on daily lifestyle habits, screen time, sleep, physical activity, and academic context.

Powered by a **FastAPI** backend and a trained **Scikit-Learn Random Forest Regressor** pipeline (`Mental_Health_Model.pkl`).

---

## 🚀 Quick Start

### 1. Requirements & Installation

Make sure you have Python 3.10+ installed. Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Run the Application

You can start the FastAPI backend server using either of the following methods:

**Method A: Using the Launcher Script**
```bash
python run_server.py
```

**Method B: Using Uvicorn Directly**
```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

**Method C: Windows Batch File**
Double-click `start_app.bat`.

---

## 🌐 Access Points

Once the server is running, open your browser:

- **Web Application UI**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Interactive Swagger API Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc API Documentation**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **Health Check API**: [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)
- **Metadata API**: [http://127.0.0.1:8000/api/metadata](http://127.0.0.1:8000/api/metadata)

---

## 🔌 API Endpoints Reference

### 1. Health Check
`GET /api/health`
Returns system status and verifies that the ML model is loaded and ready.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0",
  "model_type": "Pipeline"
}
```

---

### 2. Metadata & Form Options
`GET /api/metadata`
Returns categorical options (platforms, purposes, countries), numerical slider ranges, dataset averages, and default values.

---

### 3. Predict Score & What-If Recommendations
`POST /api/predict`
Accepts student habit features, predicts the wellbeing score (0 to 10), and computes counterfactual "What-If" habit adjustments in a single optimized batch prediction.

**Request Body:**
```json
{
  "Study_Hours": 2.8,
  "Age": 21,
  "Avg_Daily_Usage_Hours": 5.0,
  "Daily_Unlocks": 170,
  "Physical_Activity_Hours": 1.7,
  "Sleep_Hours_Per_Night": 6.6,
  "Stress_Level": "Medium",
  "Gender": "Female",
  "Academic_Level": "Undergraduate",
  "Most_Used_Platform": "Instagram",
  "Purpose_Of_Use": "Entertainment",
  "Grouped_country": "India"
}
```

**Response:**
```json
{
  "score": 6.26,
  "band": "Holding steady",
  "diff_from_mean": 0.03,
  "dataset_mean": 6.23,
  "what_if": [
    {
      "label": "Use social media one hour less",
      "delta": 0.37,
      "new_score": 6.63,
      "direction": "pos"
    },
    {
      "label": "Sleep one more hour",
      "delta": 0.2,
      "new_score": 6.46,
      "direction": "pos"
    },
    {
      "label": "Move half an hour more a day",
      "delta": -0.02,
      "new_score": 6.24,
      "direction": "zero"
    },
    {
      "label": "Unlock your phone 30 fewer times",
      "delta": -0.37,
      "new_score": 5.89,
      "direction": "neg"
    }
  ],
  "status": "success"
}
```

---

### 4. What-If Analysis Only
`POST /api/what-if`
Returns only the list of isolated habit changes and their expected delta impacts.

---

## 🧪 Testing

Automated pytest test suite covers health checks, metadata, valid/invalid inputs, edge cases, what-if counterfactual calculations, and static asset serving.

To run tests:
```bash
python -m pytest test_backend.py -v
```

---

## 📁 Project Structure

```
Mental_Health_Score_Prediction/
├── Mental_Health_Model.pkl              # Trained scikit-learn Pipeline
├── Mental_Health_Score_Prediction.ipynb # Model training & EDA notebook
├── main.py                             # FastAPI application & ML inference logic
├── run_server.py                       # Server launcher script
├── start_app.bat                       # One-click Windows starter
├── test_backend.py                     # Pytest suite
├── requirements.txt                    # Project Python dependencies
├── index.html                          # Frontend UI
├── style.css                           # Modern stylesheet & responsive layout
├── script.js                           # Frontend controller with FastAPI integration
└── README.md                           # Documentation
```
