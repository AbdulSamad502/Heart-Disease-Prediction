import pandas as pd 
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

# ----------------- Load Data -----------------
data = pd.read_csv("heartd.csv")

# ----------------- Basic Cleaning -----------------
def basic_cleaning(df):
    df = df.copy()
    df.drop("id", inplace=True, axis=1)
    df["Age_years"] = np.floor(df["age"]/365).astype(int)
    df.drop("age", inplace=True, axis=1)
    df["gender"] = df['gender'].map({1:0, 2:1})  # 0=female, 1=male
    df["BMI"] = df["weight"] / (df["height"]/100)**2
    return df

data = basic_cleaning(data)

# Fix blood pressure swaps
wrong_bp = data["ap_hi"] < data["ap_lo"]
data.loc[wrong_bp, ["ap_hi", "ap_lo"]] = data.loc[wrong_bp, ["ap_lo", "ap_hi"]].values

# ----------------- Remove/Fix Wrong Data -----------------
def removing_wrong_data(df, use_iqr=True):
    df = df.copy()
    counts = {
        "age_out_of_range": (~df["Age_years"].between(10,100)).sum(),
        "height_out_of_range": (~df["height"].between(100,250)).sum(),
        "weight_out_of_range": (~df["weight"].between(25,300)).sum(),
        "ap_hi_out_of_range": (~df["ap_hi"].between(60,300)).sum(),
        "ap_lo_out_of_range": (~df["ap_lo"].between(30,200)).sum(),
        "BMI_out_of_range": (~df["BMI"].between(10,60)).sum()
    }
    
    print("Counts of implausible values BEFORE clipping:")
    for k, v in counts.items():
        print(f"{k}: {v}")
    
    # Clip implausible values
    df["Age_years"] = df["Age_years"].clip(10,100)
    df["height"]    = df["height"].clip(100,250)
    df["weight"]    = df["weight"].clip(25,300)
    df["ap_hi"]     = df["ap_hi"].clip(60,300)
    df["ap_lo"]     = df["ap_lo"].clip(30,200)
    df["BMI"]       = df["BMI"].clip(10,60)
    
    # Optional: IQR outlier reporting
    if use_iqr:
        numeric_cols = ["Age_years","height","weight","ap_hi","ap_lo","BMI"]
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5*IQR
            upper = Q3 + 1.5*IQR
            outliers = df[(df[col] < lower) | (df[col] > upper)]
            print(f"{col} IQR outliers: {len(outliers)}")
    
    return df

data = removing_wrong_data(data)

# ----------------- Feature Engineering -----------------
data["Pulse_pressure"] = data["ap_hi"] - data["ap_lo"]
data["Mean_BP"] = (data["ap_hi"] + data["ap_lo"]) / 2

# Age groups
bins = [10,29,44,59,100]
labels = ["10-29","30-44","45-59","60+"]
data["Age_group"] = pd.cut(data["Age_years"], bins=bins, labels=labels)

# Fill missing categorical values
data["gluc"] = data["gluc"].fillna(1)
data["cholesterol"] = data["cholesterol"].fillna(1)

# Overweight binary
data["Over_weight"] = (data["BMI"] >= 25).astype(int)

# Interaction features
data["Age_bmi"] = data["Age_years"] * data["BMI"]
data["Smoke_age"] = data["Age_years"] * data["smoke"]
data["Active_bmi"] = data["active"] * data["BMI"]

print("Data cleaning & feature engineering complete!")

# ----------------- Train/Test Split -----------------
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in split.split(data, data["cardio"]):
    strat_train_set = data.loc[train_idx]
    strat_test_set = data.loc[test_idx]

train = strat_train_set.reset_index(drop=True).copy()
test = strat_test_set.reset_index(drop=True).copy()

# ----------------- Prepare Pipelines -----------------
X_train = train.drop("cardio", axis=1)
y_train = train["cardio"]

# Numerical and categorical columns
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

# Pipelines
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(sparse_output=False, drop="first"))
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# Fit and transform training data
X_train_processed = full_pipeline.fit_transform(X_train)

# Column names after one-hot encoding
cat_ohe_cols = full_pipeline.named_transformers_['cat']['encoder'].get_feature_names_out(cat_cols)
all_cols = num_cols + list(cat_ohe_cols)
X_train_processed = pd.DataFrame(X_train_processed, columns=all_cols)

# ----------------- Train Logistic Regression -----------------
lin_clf = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
lin_clf.fit(X_train_processed, y_train)

# Cross-validation on training data
cv_scores = cross_val_score(lin_clf, X_train_processed, y_train, cv=5, scoring='accuracy')
print("Cross-Validation Accuracy on training set:", cv_scores.mean())

# ----------------- Prepare Test Set -----------------
X_test = test.drop("cardio", axis=1)
y_test = test["cardio"]

X_test_processed = full_pipeline.transform(X_test)
X_test_processed = pd.DataFrame(X_test_processed, columns=all_cols)

# ----------------- Evaluate on Test Set -----------------
y_pred = lin_clf.predict(X_test_processed)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, lin_clf.predict_proba(X_test_processed)[:,1])

print(f"Test Accuracy: {acc:.4f}")
print(f"Test Precision: {prec:.4f}")
print(f"Test Recall: {rec:.4f}")
print(f"Test F1-score: {f1:.4f}")
print("Confusion Matrix:\n", cm)
print(f"ROC-AUC: {roc_auc:.4f}")

# ----------------- ROC Curve -----------------
fpr, tpr, thresholds = roc_curve(y_test, lin_clf.predict_proba(X_test_processed)[:,1])
plt.plot(fpr, tpr, label=f'ROC curve (AUC={roc_auc:.2f})')
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# ----------------- Precision-Recall Curve -----------------
prec_vals, rec_vals, _ = precision_recall_curve(y_test, lin_clf.predict_proba(X_test_processed)[:,1])
plt.plot(rec_vals, prec_vals)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()
