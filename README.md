# 📈 Revenue AI — Revenue Forecasting & Anomaly Detection System

> 🚧 **Project Status: Work in Progress**
>
> This project is currently under active development. Features, architecture, API responses, models, and the user interface may change as the system continues to improve.

An end-to-end **AI-powered Revenue Forecasting and Anomaly Detection System** built with **Python, Machine Learning, FastAPI, and Streamlit**.

The system predicts future revenue using business-related inputs such as the prediction date, number of products sold, orders, and customers. In addition to forecasting revenue, the system analyzes predictions to determine whether the revenue behavior is **Normal** or **Anomalous**.

The project follows a modern client-server architecture where the machine-learning backend is separated from the interactive frontend dashboard.

---

## 🚀 Project Overview

Revenue forecasting helps businesses make better data-driven decisions. Accurate revenue predictions can support:

* 📊 Sales planning
* 📦 Inventory management
* 📈 Business forecasting
* 💰 Revenue monitoring
* 👥 Resource allocation
* 🔍 Performance analysis
* 🚨 Early detection of unusual revenue behavior

Users enter business information through the Streamlit dashboard. The frontend sends this information to the FastAPI backend through a REST API.

The backend processes the data, generates a revenue prediction, performs anomaly detection, and returns the results to the dashboard.

---

## ✨ Features

### 📊 Revenue Forecasting

Predict expected revenue using business-related inputs, including:

* 📅 Prediction Date
* 📦 Number of Products Sold
* 🛒 Number of Orders
* 👥 Number of Customers

The machine-learning model processes these inputs and generates a predicted revenue value.

---

### 🚨 Anomaly Detection

The system analyzes the predicted revenue and determines whether the behavior is:

* 🟢 **Normal**
* 🔴 **Anomalous**

An anomaly score is also returned by the backend and displayed in the Streamlit dashboard.

---

### 🔌 FastAPI Backend

The backend exposes REST API endpoints for communication between the machine-learning system and the frontend.

Current functionality includes:

* 🩺 API health monitoring
* 📈 Revenue prediction
* 🚨 Anomaly detection
* 🤖 Model information

---

### 🎨 Interactive Streamlit Dashboard

The frontend provides an interactive interface for working with the Revenue AI system.

Dashboard features include:

* 📝 Revenue prediction form
* 💰 Predicted revenue display
* 🚨 Anomaly status
* 📊 Anomaly score visualization
* 📜 Prediction history
* 📈 Revenue history visualization
* 🩺 API health monitoring
* 📥 CSV export
* 🔎 Raw backend response viewer

---

### 📈 Prediction History

The dashboard maintains a history of predictions during the active Streamlit session.

Users can review previous predictions and visualize historical revenue predictions directly from the dashboard.

---

### 📥 CSV Export

Prediction data can be exported as CSV files.

Supported exports include:

* Individual prediction results
* Complete prediction history

This allows users to perform additional analysis outside the application.

---

### 🩺 API Health Monitoring

The Streamlit dashboard can check the availability of the FastAPI backend.

The health monitoring system can display information such as:

* API Status
* Model Status
* Model Version
* Backend URL

---

## 🏗️ System Architecture

```text
                    ┌─────────────────────────┐
                    │          USER           │
                    │                         │
                    │ Enters Business Inputs  │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │     STREAMLIT APP       │
                    │        Frontend         │
                    │                         │
                    │  • Input Form           │
                    │  • Dashboard            │
                    │  • Charts               │
                    │  • History              │
                    │  • CSV Export           │
                    └────────────┬────────────┘
                                 │
                                 │ HTTP Request
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │       FASTAPI API       │
                    │        Backend          │
                    │                         │
                    │  • /predict             │
                    │  • /health              │
                    │  • Model Information    │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   MACHINE LEARNING      │
                    │         MODEL           │
                    │                         │
                    │  Revenue Prediction     │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │    ANOMALY DETECTION    │
                    │                         │
                    │  • Normal               │
                    │  • Anomalous            │
                    │  • Anomaly Score        │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │      JSON RESPONSE      │
                    │                         │
                    │  • Predicted Revenue    │
                    │  • Anomaly Status       │
                    │  • Anomaly Score        │
                    │  • Model Information    │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   STREAMLIT DASHBOARD   │
                    │                         │
                    │ Results + Visualization │
                    └─────────────────────────┘
```

---

## 🔄 Application Workflow

The system follows the workflow below:

```text
User Input
    ↓
Streamlit Dashboard
    ↓
REST API Request
    ↓
FastAPI Backend
    ↓
Machine Learning Prediction
    ↓
Anomaly Detection
    ↓
JSON Response
    ↓
Streamlit Visualization
```

---

## 🛠️ Technology Stack

| Technology          | Purpose                        |
| ------------------- | ------------------------------ |
| 🐍 Python           | Core programming language      |
| 🤖 Machine Learning | Revenue prediction             |
| ⚡ FastAPI           | Backend REST API               |
| 🎨 Streamlit        | Interactive frontend dashboard |
| 📊 Pandas           | Data processing                |
| 🔢 NumPy            | Numerical computation          |
| 🤖 Scikit-learn     | Machine learning pipeline      |
| 📈 Plotly           | Interactive data visualization |
| 🔌 REST API         | Frontend-backend communication |

---

## 📂 Project Structure

```text
Revenue-AI-Revenue-Forecasting-Anomaly-Detection-System/
│
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   │
│   └── model/
│       └── machine_learning_model.pkl
│
├── frontend.py
│
├── requirements.txt
│
├── README.md
│
└── .gitignore
```

> The project structure may change as development continues.

---

## 🎯 Core Components

### 🤖 Machine Learning Model

The machine-learning component is responsible for generating revenue predictions based on business input data.

The model is managed by the backend and is not loaded directly by the Streamlit frontend.

---

### 🚨 Anomaly Detection System

The anomaly detection component evaluates revenue predictions and determines whether the predicted behavior appears normal or unusual.

The result is returned through the API and displayed directly in the dashboard.

---

### 🔌 REST API

The REST API connects the frontend and backend.

This separation allows the machine-learning backend and the user interface to remain independent and scalable.

---

### 🎨 Frontend Dashboard

The Streamlit dashboard acts as a client for the FastAPI backend.

The frontend:

* Collects user input
* Sends API requests
* Validates API responses
* Displays prediction results
* Visualizes data
* Maintains session-level prediction history

The frontend does **not** train or load the machine-learning model.

---

## 🚧 Project Status

### ⚠️ Work in Progress

**Revenue AI is currently under active development.**

The project is still being improved, and several areas may change in future updates, including:

* 🤖 Machine learning model improvements
* 📊 Forecasting accuracy
* 🚨 Anomaly detection logic
* 🎨 Dashboard design
* 📈 Data visualizations
* 🔌 API functionality
* 📂 Project structure
* 📱 User experience

New features and improvements will be added as development continues.

---

## 🔮 Planned Improvements

Future development may include:

* 📅 Advanced multi-day revenue forecasting
* 🤖 Improved machine-learning models
* 🚨 Advanced anomaly detection
* 📊 Additional business analytics
* 📈 More interactive visualizations
* 🔐 Authentication and user accounts
* 🗄️ Database integration
* 🐳 Docker deployment
* ☁️ Cloud deployment
* 🔄 Automated model retraining
* 📡 Real-time prediction monitoring

---

## 👨‍💻 Author

**Safi Hamid**

Aspiring Data Scientist and Machine Learning Developer.

---

## 📌 Note

> 🚧 **This project is currently a work in progress.**
>
> The application is under active development, and some features may be incomplete, experimental, or subject to change. The goal is to continuously improve the forecasting system, anomaly detection logic, backend API, and interactive dashboard.

⭐ **If you find this project interesting, consider starring the repository and following its development.**
