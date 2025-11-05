# 💳 Credit Card Fraud Detection using Machine Learning

This project detects fraudulent credit card transactions using **Machine Learning** models like **XGBoost**, **Random Forest**, and **Logistic Regression**.  
It demonstrates a complete **end-to-end Data Science workflow** — from data preprocessing to real-time web deployment using **HuggingFace**.

---

## 🚀 Live Demo  
👉 [Click here to try the app on Hugging Face](https://huggingface.co/spaces/HarshilUndhad/credit-card-fraud-detection )


---

## 📘 Project Overview

Financial fraud is a major issue for banks and card issuers.  
This project builds an ML model capable of **identifying fraudulent transactions** from highly imbalanced data.  

The app predicts whether a transaction is **Fraudulent (1)** or **Legitimate (0)** based on anonymized features.

---

## 🧩 Features

- ✅ Complete Data Science pipeline:
  - Data Cleaning & Preprocessing  
  - Feature Scaling using StandardScaler  
  - Balancing data with **SMOTE**  
  - Model Training & Evaluation  
  - Saving models with **Joblib**
- 🧠 ML Algorithms: Logistic Regression, Random Forest, XGBoost
- 🌐 Interactive **HuggingFace Web App**
- 💡 Real-time transaction prediction with confidence meter and risk visualization
- 🎨 Clean and professional UI with color-coded fraud risk levels

---

## 🧠 Tech Stack

| Category | Tools Used |
|-----------|-------------|
| **Language** | Python |
| **ML Libraries** | Scikit-learn, XGBoost, imbalanced-learn |
| **Visualization** | Matplotlib, Seaborn |
| **Deployment** | HuggingFace |
| **Data Handling** | Pandas, NumPy |

---

## 📊 Model Performance Summary

| Model | Precision | Recall | F1-Score | ROC-AUC | Accuracy |
|--------|------------|--------|----------|----------|-----------|
| Logistic Regression | 0.95 | 0.95 | 0.95 | 0.947 | 95% |
| Random Forest | 0.95 | 0.95 | 0.95 | 0.947 | 95% |
| **XGBoost (Final Model)** | **1.00** | **1.00** | **1.00** | **0.9996** | **100%** |

✅ **Final Model:** XGBoost  
✅ **Deployed Using:** HuggingFace Space  
✅ **Key Metric:** ROC-AUC = 0.9996  

---

## 🧾 How to Run Locally

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/15-13/credit-card-fraud-detection.git
cd credit-card-fraud-detection
