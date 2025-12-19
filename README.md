# 📈 Stock & Commodity Price Prediction using Predictive Analytics

This repository contains an educational predictive analytics project that applies machine learning and deep learning techniques to forecast the next trading day’s closing price for multiple stock indices and commodities using historical time-series data.

⚠️ This project is strictly for academic and analytical purposes and does not provide financial or investment advice.

---

## 🔍 Project Overview

The objective of this project is to demonstrate an end-to-end predictive analytics workflow, including data preprocessing, exploratory data analysis (EDA), feature engineering, model development, evaluation, and comparison. The system supports multiple financial instruments and allows experimentation with different predictive models to understand their performance on real-world market data.

---

## 📊 Datasets Used

Historical end-of-day futures data (01 July 2021 – 20 January 2025) for:

- BankNifty  
- Nifty  
- Gold  
- Silver  
- Crude Oil  
- Natural Gas  

Dataset attributes:



The closing price is used as the target variable for prediction.

---

## 🧠 Methodology

### Dataset Preprocessing
- Datetime parsing and chronological sorting  
- Data integrity checks  
- Time-series–safe Min–Max normalization  
- Lag-based sequence generation for supervised learning  

### Exploratory Data Analysis (EDA)
- Trend and volatility analysis  
- Instrument-wise behavior comparison  
- Statistical summaries and visualizations  

EDA notebooks are provided as exported HTML showcase files.

---

## 🤖 Models Implemented

The following predictive models were trained and evaluated:

1. Linear Regression (Baseline and best-performing model)  
2. Polynomial Regression (Degree 2)  
3. Random Forest Regressor  
4. LSTM Neural Network (TensorFlow – Experimental)  

---

## 📏 Model Evaluation

Models were evaluated using standard regression metrics:
- Mean Absolute Error (MAE)  
- Mean Squared Error (MSE)  
- Root Mean Squared Error (RMSE)  
- R-squared (R²)  

Best Model Identified:
➡️ Linear Regression, based on the lowest RMSE across multiple instruments.

---

## 🖥️ Application Features

- Instrument selection (stocks and commodities)  
- Model selection (classical ML or LSTM)  
- Next-day closing price prediction  
- Performance metric display  
- HTML-based EDA showcase  

---

## 📁 Project Structure

project/
├── data/ # CSV datasets
├── notebooks/ # Jupyter notebooks (EDA & analysis)
├── app.py # Main application
├── requirements.txt
└── README.md


---

## 🚀 How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt

python app.py
