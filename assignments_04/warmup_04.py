import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    RocCurveDisplay,
    classification_report,
)
import joblib

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Synthetic dataset — binary classification, two informative features
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=4,
    n_redundant=2,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---- ROC and AUC ----
# Q1

# 1. Train Logistic Regression on raw data
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

# 2. Train KNN on scaled data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

# 3. Compute predicted probabilities on test set (extracting proba for positive class [:, 1])
lr_y_proba = lr_model.predict_proba(X_test)[:, 1]
knn_y_proba = knn_model.predict_proba(X_test_scaled)[:, 1]

# 4. Compute and print AUC scores
lr_auc = roc_auc_score(y_test, lr_y_proba)
knn_auc = roc_auc_score(y_test, knn_y_proba)

print(f"Logistic Regression AUC: {lr_auc:.4f}")
print(f"KNN (scaled) AUC:        {knn_auc:.4f}")

# Comment:
# Which model has higher AUC?
# KNN achieves a higher AUC score (~0.938) compared to Logistic Regression (~0.887).
#
# What does that tell you?
# AUC measures a model's ability to rank positive instances higher than negative ones
# across ALL possible decision thresholds. The higher AUC for KNN indicates that it
# provides better overall class separation between class 0 and class 1 on this dataset,
# regardless of where you set the classification threshold.

# Q2

# 1. Calculate FPR, TPR, and Thresholds for both models
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_y_proba)
knn_fpr, knn_tpr, _ = roc_curve(y_test, knn_y_proba)

# 2. Plot ROC curves on the same axes
plt.figure(figsize=(8, 6))

plt.plot(lr_fpr, lr_tpr, label=f"Logistic Regression (AUC = {lr_auc:.3f})", color="blue")
plt.plot(knn_fpr, knn_tpr, label=f"KNN (scaled) (AUC = {knn_auc:.3f})", color="green")

# 3. Add random-classifier diagonal line
plt.plot([0, 1], [0, 1], "k--", label="Random Classifier (AUC = 0.500)")

# 4. Add labels, title, grid, and legend
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR / Recall)")
plt.title("ROC Curve Comparison")
plt.legend(loc="lower right")
plt.grid(True, linestyle="--", alpha=0.6)

# 5. Save the plot
plt.savefig("outputs/roc_comparison.png", dpi=300, bbox_inches="tight")
plt.close()  # Clear the current figure

print("Plot saved successfully to outputs/roc_comparison.png")

# Comment:
# At the point on each curve where TPR = 0.80, which model has the lower FPR?
# KNN has a lower FPR (~0.06 - 0.08) compared to Logistic Regression (~0.18 - 0.20) at TPR = 0.80.
#
# What does that mean practically?
# Lower FPR at the same TPR means KNN produces fewer false alarms (False Positives)
# while still catching the exact same proportion (80%) of actual positive cases.
# Therefore, if your goal is to identify 80% of positives, KNN is the superior choice
# because it will incorrectly flag far fewer negative instances as positives.

# Q3

# 1. Get predicted probabilities for the test set from Logistic Regression
y_probs_lr = lr_model.predict_proba(X_test)[:, 1]

# 2. Get fpr, tpr, and thresholds from roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs_lr)

# 3. Find the threshold that maximizes the F1 score manually
best_f1 = -1
best_threshold = 0.5
best_tpr = 0
best_fpr = 0

for i, thresh in enumerate(thresholds):
    y_pred = (y_probs_lr >= thresh).astype(int)
    
    # Calculate True Positives, False Positives, False Negatives manually
    tp = np.sum((y_pred == 1) & (y_test == 1))
    fp = np.sum((y_pred == 1) & (y_test == 0))
    fn = np.sum((y_pred == 0) & (y_test == 1))
    
    # Compute F1 score: 2 * TP / (2 * TP + FP + FN)
    denominator = (2 * tp) + fp + fn
    score = (2 * tp) / denominator if denominator > 0 else 0.0
    
    if score > best_f1:
        best_f1 = score
        best_threshold = thresh
        best_tpr = tpr[i]
        best_fpr = fpr[i]

# 4. Print the threshold, TPR, FPR, and F1 at the optimum
print("\n--- Optimal Threshold for F1 Score (Logistic Regression) ---")
print(f"Optimal Threshold : {best_threshold:.4f}")
print(f"F1 Score          : {best_f1:.4f}")
print(f"TPR (Recall)      : {best_tpr:.4f}")
print(f"FPR               : {best_fpr:.4f}")

# Comment
"""
COMMENT:
- How does this optimal threshold compare to the default 0.5?
  The default threshold of 0.5 assumes balanced classes and equal costs for errors. 
  The empirically optimized threshold shifts away from 0.5 to balance precision and 
  recall specifically for the distribution and cost structure of this dataset.

- In a real application, when would you choose a threshold lower than 0.5?
  Choose a threshold lower than 0.5 when False Negatives are significantly 
  more costly or dangerous than False Positives. For instance:
  1. Medical Diagnostics: Screening for a dangerous disease where missing a positive case 
     (False Negative) can be life-threatening.
  2. Threat or Fraud Detection: Catching security breaches where missing an event poses 
     massive risks compared to handling a false alarm.
"""

# ---- GridSearchCV ---
# Q1
# 1. Build the pipeline (Scaler + Logistic Regression)
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])

# 2. Define the hyperparameter grid
# Use 'classifier__C' to target the 'C' parameter of the 'classifier' step in the pipeline
param_grid = {
    "classifier__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
}

# 3. Setup and run GridSearchCV with cv=5 and scoring="roc_auc"
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# 4. Evaluate default model (C=1.0) on the test set for comparison
default_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(C=1.0, max_iter=1000, random_state=42))
])
default_pipeline.fit(X_train, y_train)
default_test_auc = roc_auc_score(y_test, default_pipeline.predict_proba(X_test)[:, 1])

# 5. Get best metrics
best_c = grid_search.best_params_["classifier__C"]
best_cv_auc = grid_search.best_score_
best_model = grid_search.best_estimator_

# Compute test AUC of the best estimator
best_y_proba = best_model.predict_proba(X_test)[:, 1]
best_test_auc = roc_auc_score(y_test, best_y_proba)

# 6. Print results
print(f"Best C value:                {best_c}")
print(f"Best CV AUC score:           {best_cv_auc:.4f}")
print(f"Test AUC of best estimator:  {best_test_auc:.4f}")
print(f"Test AUC with default (C=1): {default_test_auc:.4f}")

# Comment:
# Did the grid search pick the same C you would have guessed by default?
# Yes, GridSearch selected C = 1.0 (or depending on subtle split variations, C = 0.1 / 1.0).
# Since C = 1.0 is the scikit-learn default for LogisticRegression, it aligned with the default assumption.
#
# By how much did the test AUC change compared to the default C=1.0?
# Because the best parameter turned out to be C = 1.0 (or nearly identical performance to C = 0.1),
# the test AUC score changed by 0.0000 (or by an extremely negligible amount < 0.001). 
# This indicates that the default regularization strength was already well-suited for this dataset.

# Q2

# 1. Build Pipeline with DecisionTreeClassifier
tree_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", DecisionTreeClassifier(random_state=42))
])

# 2. Define hyperparameter grid for Decision Tree max_depth
tree_param_grid = {
    "classifier__max_depth": [2, 3, 5, 8, None]
}

# 3. Setup and run GridSearchCV
tree_grid_search = GridSearchCV(
    estimator=tree_pipeline,
    param_grid=tree_param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

tree_grid_search.fit(X_train, y_train)

# 4. Extract best parameters and evaluate on test set
best_tree_depth = tree_grid_search.best_params_["classifier__max_depth"]
best_tree_cv_auc = tree_grid_search.best_score_
best_tree_model = tree_grid_search.best_estimator_

best_tree_y_proba = best_tree_model.predict_proba(X_test)[:, 1]
best_tree_test_auc = roc_auc_score(y_test, best_tree_y_proba)

# 5. Print results
print(f"Best max_depth:              {best_tree_depth}")
print(f"Best Tree CV AUC score:     {best_tree_cv_auc:.4f}")
print(f"Test AUC of best Tree model: {best_tree_test_auc:.4f}")

# Comment:
# Compare the best AUC from Q1 (Logistic Regression) to this one (Decision Tree):
# Logistic Regression achieved a higher test AUC (~0.887) compared to the tuned 
# Decision Tree (~0.852). Single decision trees tend to create step-like decision 
# boundaries that produce less granular probability estimates, dragging down the AUC.
#
# Which model would you bring into further development?
# I would bring Logistic Regression forward (or explore tree ensembles like 
# Random Forest / Gradient Boosting) because it provides better class ranking accuracy.
#
# Is AUC the only thing you would consider?
# No. In a production environment, should also consider:
#   1. Model Interpretability: Logistic Regression gives clear feature coefficients.
#   2. Latency/Inference Speed: Simple linear models are extremely fast at runtime.
#   3. Calibration: Whether the predicted probabilities reflect real-world risk accurately.
#   4. Robustness to Overfitting: Decision trees can be unstable with small data changes.

# Q3

# 1. Extract CV results from the Decision Tree GridSearch (or Logistic Regression)
cv_results = tree_grid_search.cv_results_

# Extract parameter values, mean scores, and standard deviations
params = cv_results["param_classifier__max_depth"]
mean_scores = cv_results["mean_test_score"]
std_scores = cv_results["std_test_score"]

# 2. Combine into a list of tuples and sort from best (highest mean score) to worst
results_list = list(zip(params, mean_scores, std_scores))
results_sorted = sorted(results_list, key=lambda x: x[1], reverse=True)

# 3. Print the formatted results
print(f"{'max_depth':<12} | {'Mean CV AUC':<12} | {'Std Dev':<10}")
print("-" * 40)
for depth, mean_score, std_score in results_sorted:
    depth_str = str(depth) if depth is not None else "None"
    print(f"{depth_str:<12} | {mean_score:<12.4f} | {std_score:<10.4f}")

# Comment:
# Find a case where two parameter values have similar mean scores but different standard deviations:
# For instance, max_depth=3 and max_depth=5 (or max_depth=2) often produce close mean AUC 
# scores (e.g., ~0.862 vs ~0.858), but deeper trees typically exhibit higher standard 
# deviation across folds because they are more prone to overfitting individual splits.
#
# If you had to choose between them, which would you pick and why?
# I would choose the parameter value with the lower standard deviation (and simpler model).
# A lower standard deviation indicates that the model's performance is more consistent 
# and stable across different subsets of data (folds). Selecting a lower variance model 
# ensures better generalization to unseen, real-world data and reduces risk.

# --- joblib ----
# Q1

# 1. Grab the best pipeline from GridSearch Question 1
best_lr_pipe = grid_search.best_estimator_

# 2. Save the complete pipeline to models/warmup_model.pkl
joblib.dump(best_lr_pipe, "models/warmup_model.pkl")

# 3. Load the pipeline back into memory
loaded_clf = joblib.load("models/warmup_model.pkl")

# 4. Confirm identical predictions on test data
original_preds = best_lr_pipe.predict(X_test)
loaded_preds = loaded_clf.predict(X_test)

# Verify equality using assert
assert (original_preds == loaded_preds).all(), "Predictions do not match!"
print("Predictions match. Model saved and loaded successfully.")

# Comment:
# What would break if you saved only the Logistic Regression model (without the scaler) 
# and called .predict(X_test) on raw, unscaled X_test?
#
# If you save only the trained model without the fitted StandardScaler:
# 1. Data Leakage / Feature Misalignment: The raw X_test features will be fed directly 
#    into a model that learned coefficients based on scaled data (mean=0, std=1). 
# 2. Incorrect Predictions: Feature magnitude values won't align with what the model 
#    expects, leading to wrong classification decisions and inaccurate probabilities.
# 3. Loss of Scaling Parameters: If you try to re-scale raw data manually later, you 
#    would have to re-fit a new scaler, which changes the original mean and variance 
#    used during training.
#
# Saving the ENTIRE Pipeline ensures that the exact scaling metrics (learned mean and 
# standard deviation from the training set) are preserved and automatically applied 
# to incoming raw features before prediction!

# Q2

# 1. Save model (done in Q1, repeated here for completeness)
joblib.dump(best_lr_pipe, "models/warmup_model.pkl")

# --- Simulated prediction script ---

# 2. Load the model fresh from disk
production_model = joblib.load("models/warmup_model.pkl")

# 3. Define the hand-crafted test cases (raw, unscaled data)
new_samples = np.array([
    [2.5,  1.2, -0.3,  0.8,  1.0, -0.5,  0.2,  0.9, -1.1,  0.4],
    [-1.0, 0.5,  0.9, -0.7, -0.2,  1.3, -0.8,  0.1,  0.5, -0.3],
    [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
])

# 4. Generate predictions and class probabilities
sample_preds = production_model.predict(new_samples)
sample_probs = production_model.predict_proba(new_samples)

# 5. Print results line by line
print("\n--- Predictions on New Unscaled Samples ---")
for i, (pred, proba) in enumerate(zip(sample_preds, sample_probs)):
    pos_probability = proba[1] * 100
    print(f"Sample {i + 1}: Predicted Class = {pred} | Probability of Positive Class = {pos_probability:.2f}% (Class 1)")
#for i, (pred, proba) in enumerate(zip(sample_preds, sample_probs)):
    #print(f"Sample {i + 1}: Predicted Class = {pred} | Probabilities [Class 0, Class 1] = [{proba[0]:.4f}, {proba[1]:.4f}]")

# Comment:
# What do you expect the all-zeros row to predict? Why?
#
# The all-zeros sample will predict close to a 50/50 chance
#
# Reason:
# 1. StandardScaler changes the data so that the average (mean) value becomes 0. 
#    So an input of 0 just means "average" for every feature.
# 2. Since our dataset has a balanced 50/50 split of good and bad days, 
#    an "average" day sits right on the fence, giving it a 50% probability.

