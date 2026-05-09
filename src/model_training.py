import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.special import expit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, 
                             recall_score, f1_score, roc_auc_score)

# ==========================================
# STEP 1: Load Data & Preprocess
# ==========================================
train_df = pd.read_csv('hw2_train_s4850.csv')
test_df = pd.read_csv('hw2_test_s4850.csv')

X_train_raw = train_df[['x1', 'x2', 'x3']]
y_train = train_df['y']
X_test_raw = test_df[['x1', 'x2', 'x3']]

# Adding Polynomial and Interaction terms (Degree 2) to capture non-linearities
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_raw)
X_test_poly = poly.transform(X_test_raw)

# Keep feature names for readability (e.g., 'x1 x2', 'x1^2')
feature_names = poly.get_feature_names_out(['x1', 'x2', 'x3'])
X_train_poly = pd.DataFrame(X_train_poly, columns=feature_names)
X_test_poly = pd.DataFrame(X_test_poly, columns=feature_names)

# ==========================================
# STEP 2: Validation Split
# ==========================================
# We split 80% for training and 20% for validation to compare models locally
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_poly, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# ==========================================
# STEP 3: Fit Models & Compare on Validation Data
# ==========================================
print("--- MODEL EVALUATION ON VALIDATION SET ---\n")

# A) Logistic Regression (Statsmodels requires explicit constant/intercept)
X_tr_const = sm.add_constant(X_tr)
X_val_const = sm.add_constant(X_val)

log_reg = sm.Logit(y_tr, X_tr_const).fit(disp=False)
val_probs_lr = log_reg.predict(X_val_const)
val_preds_lr = (val_probs_lr > 0.5).astype(int)

print("1. LOGISTIC REGRESSION (with degree-2 interactions)")
print(f"Accuracy:  {accuracy_score(y_val, val_preds_lr):.4f}")
print(f"Precision: {precision_score(y_val, val_preds_lr):.4f}")
print(f"Recall:    {recall_score(y_val, val_preds_lr):.4f}")
print(f"F1 Score:  {f1_score(y_val, val_preds_lr):.4f}")
print(f"AUC-ROC:   {roc_auc_score(y_val, val_probs_lr):.4f}")
print("Confusion Matrix (TN, FP / FN, TP):\n", confusion_matrix(y_val, val_preds_lr))
print("-" * 40)

# B) QDA (Quadratic Discriminant Analysis)
# Note: We can pass the raw features to QDA since it inherently builds quadratic boundaries, 
# but passing the poly features makes the comparison 1:1 on the same feature space.
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_tr, y_tr)
val_preds_qda = qda.predict(X_val)
val_probs_qda = qda.predict_proba(X_val)[:, 1]

print("2. QDA (Quadratic Discriminant Analysis)")
print(f"Accuracy:  {accuracy_score(y_val, val_preds_qda):.4f}")
print(f"Precision: {precision_score(y_val, val_preds_qda):.4f}")
print(f"Recall:    {recall_score(y_val, val_preds_qda):.4f}")
print(f"F1 Score:  {f1_score(y_val, val_preds_qda):.4f}")
print(f"AUC-ROC:   {roc_auc_score(y_val, val_probs_qda):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_val, val_preds_qda))
print("==========================================\n")

# ==========================================
# STEP 4: Final Predictions on Test Set (using Logistic Regression)
# ==========================================
# Fit on the FULL training dataset for maximum information
X_train_full = sm.add_constant(X_train_poly)
X_test_full = sm.add_constant(X_test_poly)

final_log_reg = sm.Logit(y_train, X_train_full).fit(disp=False)

# Extract parameters and covariance matrix of the coefficients
params = final_log_reg.params
cov_matrix = final_log_reg.cov_params()

# Calculate the log-odds (linear predictor: X * Beta)
log_odds = np.dot(X_test_full, params)

# Calculate the standard error on the log-odds scale
# SE = sqrt( diag(X * Cov_Matrix * X.T) )
variance_log_odds = (np.dot(X_test_full, cov_matrix) * X_test_full).sum(axis=1)
se_log_odds = np.sqrt(variance_log_odds)

# Compute 95% Confidence Intervals on the log-odds scale (z = 1.96)
z_score = 1.96
ci_lower_logodds = log_odds - z_score * se_log_odds
ci_upper_logodds = log_odds + z_score * se_log_odds

# Transform log-odds back to probabilities using the inverse-logit (expit) function
prob_class1 = expit(log_odds)
ci_lower = expit(ci_lower_logodds)
ci_upper = expit(ci_upper_logodds)

# ==========================================
# STEP 5: Export Results
# ==========================================
results_df = pd.DataFrame({
    'prob_class1': prob_class1,
    'ci_lower': ci_lower,
    'ci_upper': ci_upper
})

# Save to CSV using a relative path
results_df.to_csv('results.csv', index=False)
print("Predictions saved to 'results.csv' successfully.")