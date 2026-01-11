# ❤️ Cardiovascular Disease Risk Prediction (ML + Streamlit)

A professional Machine Learning project that predicts the risk of **cardiovascular disease** using clinical and lifestyle parameters.  
This application is built using **Python, Scikit-learn, Streamlit**, and a trained ML model (**Random Forest Classifier**).

---

## 📌 Project Overview

Cardiovascular diseases are one of the leading causes of death worldwide.  
This project predicts whether a person is at risk of cardiovascular disease based on health measurements such as:

- Age
- Height / Weight
- Systolic / Diastolic Blood Pressure
- Cholesterol Level
- Glucose Level
- Smoking & Alcohol intake
- Gender

The application provides a **risk percentage**, prediction output, and includes **charts and health insights** through a multi-page Streamlit UI.

---

## 🚀 Features

✅ Disease Risk Prediction using trained ML model  
✅ Risk Level shown in **percentage**  
✅ Clean and interactive **multi-page Streamlit UI**  
✅ Health dashboard + risk analytics
✅ Model accuracy : 73.5%
✅ Model evaluation table:
- Train-Test Split accuracy
- K-Fold Cross Validation accuracy
- Hyperparameter tuning accuracy  
✅ Download medical report option  

---

## 🧠 Model Details

- **Algorithm Used:** Random Forest Classifier  
- **Preprocessing:** StandardScaler + Encoding  
- **Training Approaches Used:**
  - Train-Test Split accuracy
  - K-Fold Cross Validation
  - Hyperparameter tuning (GridSearchCV)

---

## 🏗 Tech Stack

### Frontend (UI)
- Streamlit
- Plotly
- Streamlit native charts

### Backend (Model)
- Python
- Pandas / NumPy
- Scikit-learn
- Pickle (model saving)

---

## 📂 Repository Structure

```bash
Cardio_ML_Project/
│
├── app.py
├── data/
│   ├── cardio_preprocessed.csv
│   ├── cardio_train.csv
│
├── models/
│   ├── feature_columns.pkl
│   ├── scaler.pkl
│   ├── mappings.pkl
│   └── rf_model.pkl
│
├── notebooks/
│   ├── cleaning_preprocessing.ipynb
│   └── model.ipynb
│
├── requirements.txt
└── runtime.txt
