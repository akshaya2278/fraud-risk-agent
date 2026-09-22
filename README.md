# 🛡️ AI-Powered Fraud Detection & Transaction Risk Agent

An end-to-end FinTech machine learning pipeline and interactive dashboard built to detect fraudulent credit card transactions in real-time. Designed for enterprise-scale transaction processing, this agent goes beyond simple classification by providing **plain-English AI explainability** for human risk analysts.

## 🌟 Key Features

* **Dual-Engine ML Architecture:** Combines a supervised **XGBoost** classifier (trained on known fraud vectors) with an unsupervised **Isolation Forest** (to detect zero-day anomalies and unknown attack patterns).
* **Behavioral Feature Engineering:** Dynamically calculates rolling user averages, time elapsed since previous transactions, and rapid-velocity hourly frequencies to catch synthetic fraud behaviors.
* **Explainable AI (XAI):** Integrates **SHAP (SHapley Additive exPlanations)** to translate complex mathematical risk scores into human-readable insights (e.g., *"Amount is 5x higher than the user's normal average"*).
* **Interactive Analyst Dashboard:** A polished **Streamlit** frontend allowing risk teams to upload batch transaction CSVs, view hybrid risk scores, and process a prioritized queue of suspicious activity.

## 🛠️ Tech Stack
* **Language:** Python 3.11
* **Data Processing:** Pandas, NumPy, Scikit-Learn
* **Machine Learning:** XGBoost, Isolation Forest
* **Explainability:** SHAP
* **Frontend UI:** Streamlit

## 🚀 How to Run Locally

**1. Clone the repository and navigate to the project directory:**
```bash
git clone [https://github.com/akshaya2278/fraud-risk-agent.git](https://github.com/akshaya2278/fraud-risk-agent.git)
cd fraud-risk-agent
