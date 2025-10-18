# Heart Disease Prediction using Logistic Regression

## Project Overview
This project predicts the likelihood of heart disease (cardio condition) in individuals based on medical and lifestyle data. It uses Python, Pandas, NumPy, and Scikit-learn for data preprocessing, feature engineering, model training, and evaluation, with visualizations using Matplotlib.

---

## Dataset
- **Source:** `heartd.csv`
- **Description:** Contains patient-level data including:
  - Age, Gender, Height, Weight, Blood Pressure (ap_hi, ap_lo)
  - Cholesterol, Glucose, Smoking, Alcohol intake, Physical activity
  - Target variable: `cardio` (0 = no heart disease, 1 = heart disease)

---

## Features & Preprocessing
- Basic cleaning:
  - Converted age from days to years.
  - Computed BMI.
  - Corrected swapped blood pressure values.
  - Fixed implausible values using clipping and IQR.
- Feature engineering:
  - Pulse Pressure (`ap_hi - ap_lo`) and Mean BP (`(ap_hi+ap_lo)/2`)
  - Age groups (`10-29`, `30-44`, `45-59`, `60+`)
  - Overweight binary indicator
  - Interaction features (Age * BMI, Age * Smoke, Active * BMI)
- Missing categorical values imputed with the most frequent value.

---

## Modeling
- **Train/Test Split:** Stratified by target variable (`cardio`) to maintain class distribution.
- **Pipeline:** 
  - Numerical features: Imputation + StandardScaler
  - Categorical features: Imputation + OneHotEncoding
- **Model:** Logistic Regression (balanced class weights, max_iter=1000)
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix

---

## Visualizations
- ROC Curve with AUC score
- Precision-Recall Curve

---

## Key Insights
- Model performance evaluated on the test set using multiple metrics.
- Data preprocessing and feature engineering help in improving model accuracy.
- ROC-AUC provides insight into the model's discrimination ability.
- Precision-Recall curve shows performance on imbalanced classes.

---

## Technologies Used
- Python 3.x
- Pandas, NumPy
- Scikit-learn
- Matplotlib

---

## How to Run
1. Clone the repository:
```bash
git clone https://github.com/AbdulSamad/Heart-Disease-Prediction.git
cd Heart-Disease-Prediction

# Install dependencies:
pip install pandas numpy scikit-learn matplotlib

# Run the script:
python main.py

#Author

Abdul Samad
Email:-samad784600@gmail.com
GitHub: https://github.com/AbdulSamad502