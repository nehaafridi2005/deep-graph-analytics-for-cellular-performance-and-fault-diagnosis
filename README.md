# deep-graph-analytics-for-cellular-performance-and-fault-diagnosis
 📖 Overview
This project presents an intelligent system for analyzing cellular network performance and diagnosing faults using advanced machine learning techniques. The rapid growth of wireless communication technologies such as 4G, 5G, and Wi-Fi has increased the complexity of network management.
The proposed system is a **Flask-based web application** that integrates data preprocessing, exploratory data analysis (EDA), machine learning models, and real-time prediction to provide efficient and automated fault diagnosis.

 🎯 Objectives
* Analyze cellular network performance using data-driven techniques
* Predict network type and signal strength
* Detect faults and anomalies in cellular networks
* Improve network reliability and efficiency
* Provide real-time prediction and diagnosis
  
❗ Problem Statement
Modern telecom networks generate massive amounts of dynamic data, making manual fault detection inefficient. Traditional systems:
* Fail to capture complex relationships between network parameters
* Are time-consuming and error-prone
* Lack real-time responsiveness
This project aims to develop an intelligent system that can automatically analyze and predict network performance using machine learning.
---

⚙️ Technologies Used
* Python
* Flask (Web Framework)
* Scikit-learn
* Pandas, NumPy
* Matplotlib, Seaborn
* Joblib
 
 📊 Dataset
* Training Data: `signal_metrics.csv`
* Testing Data: `TestData.csv`

 Features Used:
* Signal Strength (dBm)
* Latency
* Throughput
* Signal Quality
* Network Type (Target - Classification)

🏗️ Project Architecture
The system follows a structured workflow:
1. Data Loading
2. Data Preprocessing
3. Exploratory Data Analysis (EDA)
4. Model Training
5. Model Evaluation
6. Real-Time Prediction

 🔄 Methodology

1. Data Preprocessing

* Handling missing values
* Label encoding for categorical features
* Feature scaling using StandardScaler and MinMaxScaler

2. Exploratory Data Analysis

* Distribution plots
* Correlation matrix
* Feature analysis

 3. Model Building

 Classification Models (Network Type Prediction)

* Ridge Classifier
* Decision Tree Classifier
* CatBoost Classifier

 Regression Models (Signal Strength Prediction)

* Ridge Regressor
* Decision Tree Regressor
* CatBoost Regressor

 📈 Evaluation Metrics
Classification:

* Accuracy
* Precision
* Recall
* F1-Score
 Regression:

* MAE (Mean Absolute Error)
* MSE (Mean Squared Error)
* RMSE (Root Mean Squared Error)
* R² Score

 ▶️ How to Run the Project

 Step 1: Clone Repository

```bash
git clone https://github.com/your-username/deep-graph-analytics.git
cd deep-graph-analytics
```
 Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```
 Step 3: Run Application

```bash
python app.py
```
 Step 4: Open in Browser

```
http://127.0.0.1:5000/
```
 🌐 Features

* Web-based interface using Flask
* Real-time prediction of network type and signal strength
* Visualization of network performance
* Comparison of multiple ML models
* Automated preprocessing and analysis

 📉 Existing System Limitations

* Manual fault detection
* Time-consuming analysis
* High dependency on human expertise
* Lack of scalability
* Poor real-time performance

🚀 Proposed System Advantages

* Automated fault diagnosis
* Faster and more accurate predictions
* Scalable and efficient
* Real-time analysis capability
* Reduced human intervention

🔮 Future Scope

* Integration with real-time telecom data
* Use of Graph Neural Networks (GNN)
* Deployment on cloud platforms
* Advanced deep learning models

 💻 System Requirements
 Software:
* Python 3.x
* Flask
* Jupyter Notebook (optional)

 Hardware:
* Minimum 4GB RAM
* Standard Processor
