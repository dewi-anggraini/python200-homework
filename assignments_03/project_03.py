# ---Part 2: Mini-Project -- Spam or Ham? A Classifier Shootout---
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA


# Ensure outputs directory exists
os.makedirs("outputs", exist_ok=True)

# Task 1: Load and Explore
# 1. Load the Dataset
# Source: UCI Machine Learning Repository Spambase
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
names_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.names"

# Define columns manually based on the spambase documentation
# 48 word_freq_WORD, 6 char_freq_CHAR, 3 capital_run_length_PARAMETER, 1 target
word_features = [
    "make", "address", "all", "3d", "our", "over", "remove", "internet",
    "order", "mail", "receive", "will", "people", "report", "addresses",
    "free", "business", "email", "you", "credit", "your", "font", "000",
    "money", "hp", "hpl", "george", "650", "lab", "labs", "telnet", "857",
    "data", "415", "85", "technology", "1999", "parts", "pm", "direct",
    "cs", "meeting", "original", "project", "re", "edu", "table", "conference"
]
word_cols = [f"word_freq_{w}" for w in word_features]
char_cols = [f"char_freq_{c}" for c in [";", "(", "[", "!", "$", "#"]]
capital_cols = ["capital_run_length_average", "capital_run_length_longest", "capital_run_length_total"]
feature_names = word_cols + char_cols + capital_cols
column_names = feature_names + ["spam_label"]

# Fetch and read data
print("Loading dataset...")
df = pd.read_csv(url, header=None, names=column_names)

# 2. Basic Dataset Exploration
total_emails = len(df)
class_counts = df["spam_label"].value_counts()
class_percentages = df["spam_label"].value_counts(normalize=True) * 100

print(f"\n--- Dataset Summary ---")
print(f"Total number of emails: {total_emails}")
print(f"Ham (0) count: {class_counts[0]} ({class_percentages[0]:.2f}%)")
print(f"Spam (1) count: {class_counts[1]} ({class_percentages[1]:.2f}%)")

# -----------------------------------------------------
# Dataset Balance Discussion
# -----------------------------------------------------
#
# The dataset contains more ham (non-spam) emails than
# spam emails, but the imbalance is not extreme.
#
# Because the classes are not perfectly balanced,
# accuracy alone is not enough to evaluate a classifier.
# A model could obtain a high accuracy by predicting the
# majority class more often.
#
# For this reason, precision, recall, and the F1-score
# should also be considered when evaluating spam
# detection models.
# -----------------------------------------------------

print("""
Dataset Discussion:
The dataset contains more ham emails than spam emails,
so the classes are moderately imbalanced. This means
accuracy should not be interpreted by itself because a
model could achieve a relatively high accuracy simply by
predicting the majority class. Precision, recall, and
the F1-score provide a more complete picture of model
performance.
""")

# 3. Generate and Save Boxplots
# Apply a log scale (y+1) because the distribution contains heavy right-skewed outliers
features_to_plot = ["word_freq_free", "char_freq_!", "capital_run_length_total"]

print("\nGenerating boxplots...")
for col in features_to_plot:
    plt.figure(figsize=(6, 5))
    
    # Using a log-transformed axis helps visualize the heavily skewed data clearly
    sns.boxplot(x="spam_label", y=col, hue="spam_label", data=df, palette="Set2", legend=False)
    
    plt.title(f"Distribution of {col} by Class")
    plt.xlabel("Email Label (0 = Ham, 1 = Spam)")
    plt.ylabel(col)
    
    # Adjust y-axis to log scale for extreme outliers (especially for capital runs)
    if col == "capital_run_length_total":
        plt.yscale("log")
        plt.ylabel(f"{col} (Log Scale)")
        
    plt.tight_layout()
    plt.savefig(f"outputs/{col}_boxplot.png")
    plt.close()

print("Boxplots saved successfully to the 'outputs/' directory.")

print("""
Boxplot Observations:
Spam emails generally contain higher values for
word_freq_free, char_freq_!, and
capital_run_length_total than ham emails.

Although there is some overlap between the two classes,
spam emails tend to use promotional words, excessive
punctuation, and longer runs of capital letters more
frequently. These differences suggest that these
features may help machine learning models distinguish
spam from legitimate emails.
""")

# Task 2: Prepare Your Data

# 1. Separate Features (X) and Target (y)
# -----------------------------------------------------
# CHOICE: 80/20 Train-Test Split with Stratification
#
# WHY:
# - An 80/20 split provides a large training set while reserving enough
#   unseen data for reliable evaluation.
# - Using stratify=y preserves the original spam/ham class proportions
#   in both the training and testing datasets.
# - random_state=42 ensures that the split is reproducible.
# -----------------------------------------------------

X = df.drop(columns=["spam_label"])
y = df["spam_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("\n--- Data Split Summary ---")
print(f"Training samples : {X_train.shape[0]}")
print(f"Testing samples  : {X_test.shape[0]}")
print(f"Training spam ratio : {y_train.mean():.4f}")
print(f"Testing spam ratio  : {y_test.mean():.4f}")


# -----------------------------------------------------
# 2. Feature Scaling
# -----------------------------------------------------
# CHOICE: StandardScaler
#
# WHY:
# Many machine learning algorithms (especially KNN, Logistic Regression,
# and PCA) are sensitive to the scale of the input features.
#
# In the Spambase dataset:
# - Word frequencies are usually very small percentages.
# - Capital letter statistics can have values in the thousands.
#
# Without scaling, the larger-valued features would dominate the model.

# Fit the scaler ONLY on the training data to prevent information
# from the test set leaking into the training process.
# -----------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrames to preserve feature names for debugging
X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=X.columns
)

X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=X.columns
)

print("\nFeature scaling completed successfully.")

# PCA Preprocessing
# 3. Principal Component Analysis (PCA)

# WHY:
# PCA reduces the number of features while retaining as much information
# (variance) as possible.
#
# Instead of using all 57 original features, PCA creates new features
# called principal components that summarize the important patterns
# in the data.
#
# PCA must be fitted ONLY on the training data to avoid data leakage.
# -----------------------------------------------------

pca = PCA()

# Learn the principal components from the training data only
pca.fit(X_train_scaled)

# -----------------------------------------------------
# Determine the number of components that explain
# at least 90% of the variance.
# -----------------------------------------------------

cumulative_variance = np.cumsum(
    pca.explained_variance_ratio_
)

n = np.argmax(cumulative_variance >= 0.90) + 1

print(f"\nNumber of principal components for 90% variance: {n}")
print(f"Variance retained: {cumulative_variance[n-1]:.4f}")


# -----------------------------------------------------
# Plot cumulative explained variance
# -----------------------------------------------------
# This plot shows how much information is retained as more
# principal components are included.
# -----------------------------------------------------

plt.figure(figsize=(8,5))

plt.plot(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance,
    marker="o"
)

plt.axhline(
    y=0.90,
    color="red",
    linestyle="--",
    label="90% Variance"
)

plt.axvline(
    x=n,
    color="green",
    linestyle="--",
    label=f"{n} Components"
)

plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Cumulative Explained Variance")

plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("outputs/spam_pca_explained_variance.png")
plt.close()

print("PCA explained variance plot saved to outputs/spam_pca_explained_variance.png")


# Transform the datasets using the learned PCA model

# Keep only the first n principal components because they
# preserve at least 90% of the original variance.
#
# Also keep the fully scaled datasets because later tasks
# require comparing models trained on:
#   - unscaled data
#   - scaled data
#   - PCA-reduced data

X_train_pca = pca.transform(X_train_scaled)[:, :n]
X_test_pca = pca.transform(X_test_scaled)[:, :n]

print(f"\nOriginal feature count : {X_train.shape[1]}")
print(f"PCA feature count      : {X_train_pca.shape[1]}")

print("\nTask 2 completed successfully.")


# Task 3: A Classifier Comparison

print("\n" + "="*40)
print("=== Task 3: Classifier Comparison ===")
print("="*40)
# Store model accuracy results for comparison plus easier to read
model_results = {}
# Store predictions from every model for later comparison
predictions_store = {}

# Helper function to train a model and print the evaluation results
def run_evaluation(model, train_features, test_features, model_name):
    model.fit(train_features, y_train)
    predictions = model.predict(test_features)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"\n[Classifier]: {model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, predictions))
    return accuracy, predictions


# 1. KNN trained on UNSCALED data

knn_unscaled_acc, knn_unscaled_predictions = run_evaluation(
    KNeighborsClassifier(n_neighbors=5),
    X_train,
    X_test,
    "KNN Unscaled"
)

model_results["KNN Unscaled"] = knn_unscaled_acc
predictions_store["KNN Unscaled"] = knn_unscaled_predictions


# 2. Compare KNN using scaled features and PCA-reduced features

knn_scaled_acc, knn_scaled_predictions = run_evaluation(
    KNeighborsClassifier(n_neighbors=5),
    X_train_scaled,
    X_test_scaled,
    "KNN Scaled"
)

model_results["KNN Scaled"] = knn_scaled_acc
predictions_store["KNN Scaled"] = knn_scaled_predictions

# 3. KNN PCA
knn_pca_acc, knn_pca_predictions = run_evaluation(
    KNeighborsClassifier(n_neighbors=5),
    X_train_pca,
    X_test_pca,
    "KNN PCA"
)

model_results["KNN PCA"] = knn_pca_acc
predictions_store["KNN PCA"] = knn_pca_predictions

# =====================================================
# KNN Comparison
# =====================================================

print("\n" + "="*40)
print("=== KNN: SCALED VS PCA COMPARISON ===")
print("="*40)

print("\nKNN Comparison:")
print(f"Scaled KNN Accuracy : {knn_scaled_acc:.4f}")
print(f"PCA KNN Accuracy    : {knn_pca_acc:.4f}")

difference = abs(knn_scaled_acc - knn_pca_acc)

if knn_scaled_acc > knn_pca_acc:
    print(f"""
Scaled data performed better by {difference:.4f}.
This suggests that PCA removed some information that was
useful for KNN's distance calculations.
""")
elif knn_pca_acc > knn_scaled_acc:
    print(f"""
PCA performed better by {difference:.4f}.
Reducing the dimensionality likely removed redundant
features and improved KNN's ability to generalize.
""")
else:
    print("""
Both approaches achieved the same accuracy, indicating
that PCA preserved the important information needed by
KNN.
""")


# 4. Decision Tree Classifier - Depth Tuning

print("\n" + "="*40)
print("=== DECISION TREE DEPTH TESTING ===")
print("="*40)

depth_options = [3, 5, 10, None]

depth_results = {}

for depth in depth_options:

    dt_test = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    dt_test.fit(X_train, y_train)

    train_accuracy = accuracy_score(
        y_train,
        dt_test.predict(X_train)
    )

    test_accuracy = accuracy_score(
        y_test,
        dt_test.predict(X_test)
    )

    depth_results[depth] = {
        "train": train_accuracy,
        "test": test_accuracy
    }

    print(
        f"Depth {depth}: "
        f"Train Accuracy={train_accuracy:.4f}, "
        f"Test Accuracy={test_accuracy:.4f}"
    )

best_depth_accuracy = depth_results[None]["test"]
depth10_accuracy = depth_results[10]["test"]

accuracy_difference = best_depth_accuracy - depth10_accuracy

print(f"""
Decision Tree Selection:

max_depth=None test accuracy:
{best_depth_accuracy:.4f}

max_depth=10 test accuracy:
{depth10_accuracy:.4f}

Accuracy difference:
{accuracy_difference:.4f}

Although max_depth=None achieved slightly higher accuracy,
it has a much larger gap between training and testing accuracy,
indicating overfitting.

max_depth=10 was selected because it provides similar predictive
performance with lower model complexity and better expected
generalization.
""")

# -----------------------------------------------------
# Decision Tree Model Decision:
# -- Decision Tree Depth Testing ---
# Depth 3: Train Accuracy=0.8965, Test Accuracy=0.8849
# Depth 5: Train Accuracy=0.9234, Test Accuracy=0.8990
# Depth 10: Train Accuracy=0.9674, Test Accuracy=0.9088
# Depth None: Train Accuracy=0.9997, Test Accuracy=0.9110
#
# I selected max_depth=10 for the production model.
#
# The highest test accuracy was achieved by max_depth=None
# (0.9110). However, this model also achieved almost perfect
# training accuracy (0.9997), which indicates that the tree
# learned the training data too closely and may overfit.
#
# The max_depth=10 model achieved a very similar test accuracy
# (0.9088) while having lower training accuracy (0.9674).
# The accuracy difference between the two models was only
# 0.22%, but max_depth=10 provides a simpler model with better
# balance between performance and generalization.
#
# Therefore, max_depth=10 was selected as the final Decision
# Tree configuration.
# -----------------------------------------------------

chosen_depth = 10

production_dt = DecisionTreeClassifier(
    max_depth=chosen_depth,
    random_state=42
)


dt_accuracy, dt_predictions = run_evaluation(
    production_dt,
    X_train,
    X_test,
    f"Decision Tree (max_depth={chosen_depth})"
)


model_results["Decision Tree"] = dt_accuracy
predictions_store["Decision Tree"] = dt_predictions


# 5. RandomForestClassifier

# Random Forest combines many decision trees, which usually
# reduces overfitting compared to using a single tree.
rf_model = RandomForestClassifier(
    random_state=42
)
rf_accuracy, rf_predictions = run_evaluation(
    rf_model,
    X_train,
    X_test,
    "Random Forest"
)

model_results["Random Forest"] = rf_accuracy
predictions_store["Random Forest"] = rf_predictions

# Decision Tree Feature Importances

dt_importances = pd.Series(
    production_dt.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

print("\nTop 10 Decision Tree Feature Importances")
print(dt_importances.head(10))

# Random Forest Feature Importances
rf_importances = pd.Series(
    rf_model.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

print("\nTop 10 Random Forest Feature Importances")
print(rf_importances.head(10))

# Save the required figure
plt.figure(figsize=(10,6))

rf_importances.head(10).plot(kind="bar")

plt.title("Top 10 Random Forest Feature Importances")
plt.xlabel("Feature")
plt.ylabel("Importance")

plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig("outputs/feature_importances.png")
plt.close()

print("Random Forest feature importance plot saved to outputs/feature_importances.png")

print("""
Feature Importance Discussion:

The Decision Tree and Random Forest identified many of the
same important features, although the Random Forest
distributed importance more evenly across multiple
features.

Several of the highest-ranked features are related to
words and character frequencies that are commonly found
in spam emails, which matches the intuition that spam
messages often contain promotional language and unusual
symbols.

Because the Random Forest averages many trees, its feature
importance estimates are generally more stable than those
from a single Decision Tree.
""")

# 6. LogisticRegression trained on SCALED data vs PCA-REDUCED data

lr_scaled_acc, lr_scaled_predictions = run_evaluation(
    LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="liblinear"
    ),
    X_train_scaled,
    X_test_scaled,
    "Logistic Regression Scaled"
)

model_results["Logistic Regression Scaled"] = lr_scaled_acc
predictions_store["Logistic Regression Scaled"] = lr_scaled_predictions

# 7. Logistic Regression PCA

lr_pca_acc, lr_pca_predictions = run_evaluation(
    LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="liblinear"
    ),
    X_train_pca,
    X_test_pca,
    "Logistic Regression PCA"
)

model_results["Logistic Regression PCA"] = lr_pca_acc
predictions_store["Logistic Regression PCA"] = lr_pca_predictions

# =====================================================
# Logistic Regression Comparison 
# =====================================================

print("\n" + "="*40)
print("=== LOGISTIC REGRESSION: SCALED VS PCA COMPARISON ===")
print("="*40)

print(f"Scaled Accuracy : {lr_scaled_acc:.4f}")
print(f"PCA Accuracy    : {lr_pca_acc:.4f}")

difference = abs(lr_scaled_acc - lr_pca_acc)

if lr_scaled_acc > lr_pca_acc:
    print(f"""
Scaled data performed better by {difference:.4f}.

This indicates that using the original scaled features
retained slightly more useful information than the
PCA-reduced dataset.
""")
elif lr_pca_acc > lr_scaled_acc:
    print(f"""
PCA performed better by {difference:.4f}.

Removing redundant features appears to have slightly
improved Logistic Regression's performance.
""")
else:
    print("""
Both approaches achieved the same accuracy, showing that
PCA preserved nearly all of the useful predictive
information.
""")

# Final Model Comparison and Analysis

# Display accuracy comparison for all models
print("\n" + "="*40)
print("=== FINAL MODEL PERFORMANCE COMPARISON ===")
print("="*40)

for model_name, accuracy in model_results.items():
    print(f"{model_name:<35} : {accuracy:.4f}")


# Find the model with the highest test accuracy
best_model_name = max(
    model_results,
    key=model_results.get
)

best_accuracy = model_results[best_model_name]

print("\n--- Best Performing Model ---")
print(f"Model: {best_model_name}")
print(f"Test Accuracy: {best_accuracy:.4f}")

print("""
Overall Discussion:

The Random Forest achieved the highest overall test
accuracy, making it the strongest classifier for this
dataset. Its ensemble approach combines many decision
trees, reducing overfitting while capturing complex
relationships between features.

The Decision Tree showed increasing training accuracy as
tree depth increased, while test accuracy improved only
slightly. This indicates that deeper trees began to
overfit the training data.

For KNN and Logistic Regression, comparing scaled data
with PCA-reduced data demonstrated whether reducing the
number of features improved classification performance.
The scaled feature sets
performed better for both KNN and Logistic Regression.
""")

# =====================================================
# Hypothesis Comparison Summary
# =====================================================

print("\n" + "="*40)
print("=== HYPOTHESIS COMPARISON SUMMARY ===")
print("="*40)

print("""
The results partially supported my hypothesis from Task 2.

I expected PCA to improve KNN performance by reducing
noise and removing redundant features. However, KNN
performed slightly better using scaled data (0.9077)
compared with PCA-reduced data (0.9066). This suggests
that some original features contained useful information
for KNN's distance calculations.

I also expected Logistic Regression to perform similarly
with scaled and PCA-reduced data. This was partially
supported because both approaches performed well, but
scaled data achieved higher accuracy (0.9294) compared
with PCA (0.9186).

Overall, PCA successfully reduced the number of features
from 57 to 43 while preserving 90.68% of the variance,
but it did not improve model performance. The scaled
feature sets produced better results for both KNN and
Logistic Regression.
""")

# =====================================================
# Spam Filter Discussion
# =====================================================

print("\n" + "="*40)
print("=== SPAM FILTER METRIC DISCUSSION ===")
print("="*40)

print("""
Spam Filter Discussion:

Accuracy alone is not the best metric for evaluating a
spam filter.

I would prioritize minimizing false positives because a
false positive incorrectly marks a legitimate email as
spam. Missing an important email could have more serious
consequences than receiving an unwanted spam message.

Although false negatives allow spam into the inbox,
users can usually delete those messages manually.
Therefore, precision is especially important, while
recall should also be monitored to ensure that too much
spam is not missed.

Considering accuracy together with precision, recall,
and the F1-score provides a much more complete
evaluation of a spam detection system.
""")

# =====================================================
# Confusion Matrix for Best Model
# =====================================================

print("\n" + "="*40)
print("=== CONFUSION MATRIX ANALYSIS ===")
print("="*40)


# Use predictions from the best model
# Make sure predictions are stored when each model is evaluated

best_predictions = predictions_store[best_model_name]

cm = confusion_matrix(
    y_test,
    best_predictions
)

false_positives = cm[0, 1]
false_negatives = cm[1, 0]


print(f"False Positives (Ham predicted as Spam): {false_positives}")
print(f"False Negatives (Spam predicted as Ham): {false_negatives}")


if false_positives > false_negatives:
    print("The model makes more false positive errors.")
else:
    print("The model makes more false negative errors.")


# Display and save confusion matrix

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Ham", "Spam"]
)

disp.plot(cmap=plt.cm.Blues)

plt.title(f"Confusion Matrix - {best_model_name}")

plt.tight_layout()

plt.savefig(
    "outputs/best_model_confusion_matrix.png"
)

plt.close()


print("\nConfusion matrix saved successfully.")
print("""
Confusion Matrix Interpretation:

The Random Forest model produced more false negatives
(33) than false positives (18). This means the model is
more likely to allow spam messages into the inbox than to
incorrectly block legitimate emails.

For a spam filter, this may be acceptable depending on the
goal of the system. However, if preventing unwanted spam
is the priority, improving recall for the spam class would
be important.
""")


# Task 4: Cross-Validation

print("\n" + "="*40)
print("=== STARTING 5-FOLD CROSS-VALIDATION ===")
print("="*40)

# Define the models explicitly using Pipelines where scaling/PCA is required.
# Using a Pipeline ensures that scaling and PCA are fit ONLY on the 4 internal 
# training folds and applied to the validation fold, preventing data leakage.
cv_models = {
    "1. KNN - Unscaled Data": KNeighborsClassifier(n_neighbors=5),
    
    "2. KNN - Scaled Data": Pipeline([
        ('scaler', StandardScaler()), 
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ]),
    
    "3. KNN - PCA-Reduced Data": Pipeline([
        ('scaler', StandardScaler()), 
        ('pca', PCA(n_components=n)), 
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ]),
    
    f"4. Decision Tree (max_depth={chosen_depth})": DecisionTreeClassifier(max_depth=chosen_depth, random_state=42),
    
    "5. Random Forest": RandomForestClassifier(random_state=42),
    
    "6. Logistic Regression - Scaled Data": Pipeline([
        ('scaler', StandardScaler()), 
        ('lr', LogisticRegression(C=1.0, max_iter=1000, solver='liblinear'))
    ]),
    
    "7. Logistic Regression - PCA-Reduced Data": Pipeline([
        ('scaler', StandardScaler()), 
        ('pca', PCA(n_components=n)), 
        ('lr', LogisticRegression(C=1.0, max_iter=1000, solver='liblinear'))
    ])
}

# Run 5-fold cross-validation on the raw training dataset (X_train, y_train)
# The Pipelines handles the internal scaling/PCA transforms dynamically per fold
cv_results = {}

for name, model in cv_models.items():
    # cv=5 means 5-fold cross-validation
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
    mean_score = scores.mean()
    std_score = scores.std()
    cv_results[name] = (mean_score, std_score)
    
    print(f"{name:<45} | Mean Accuracy: {mean_score:.4f} | Std Dev: {std_score:.4f}")

# Find the most accurate and most stable models programmatically
most_accurate = max(cv_results, key=lambda k: cv_results[k][0])
most_stable = min(cv_results, key=lambda k: cv_results[k][1])

print("\n--- Cross-Validation Insights Summary ---")
print(f"Most Accurate Model on CV: {most_accurate} ({cv_results[most_accurate][0]:.4f})")
print(f"Most Stable Model (Lowest Std Dev): {most_stable} (Std Dev: {cv_results[most_stable][1]:.4f})")


# VARIANCE COMPARISON: RANDOM FOREST VS DECISION TREE

dt_std = cv_results[f"4. Decision Tree (max_depth={chosen_depth})"][1]
rf_std = cv_results["5. Random Forest"][1]

print(f"\nVariance Shootout:")
print(f" -> Decision Tree Fold Std Dev: {dt_std:.4f}")
print(f" -> Random Forest Fold Std Dev: {rf_std:.4f}")

if rf_std < dt_std:
    print(" -> Observation verified: The Random Forest shows lower variance across folds than the single Decision Tree.")
else:
    print(" -> Observation: Variance levels are highly comparable between the tree structures.")

# Comment:
# The Random Forest achieved the highest average accuracy in
# both the train/test split and 5-fold cross-validation results.
# This shows that Random Forest was the best-performing model
# for this dataset.
#
# The model with the lowest standard deviation was the most
# stable because its results changed less between the five folds.
# Random Forest also showed good stability compared with the
# single Decision Tree because it combines multiple trees.
#
# The cross-validation ranking was similar to the train/test
# split ranking. This means the model comparison was consistent
# and the results were not only caused by one random data split.
#
# Overall, the cross-validation results support selecting
# Random Forest as the strongest model because it achieved high
# accuracy and reliable performance.

# Task 5: Building a Prediction Pipeline

print("\n" + "="*40)
print("=== TRAINING FINAL PRODUCTION PIPELINES ===")
print("="*40)

# 1. Best Tree-Based Pipeline (Random Forest)
# As determined in Tasks 3 and 4, Random Forest is immune to scale discrepancies.
# Therefore, it needs no preprocessing transformation steps.
tree_pipeline = Pipeline([
    ('classifier', RandomForestClassifier(random_state=42))
])

print("\nFitting Best Tree-Based Pipeline...")
tree_pipeline.fit(X_train, y_train)
tree_preds = tree_pipeline.predict(X_test)

print("\n[Pipeline Results]: Random Forest")
print(classification_report(y_test, tree_preds))


# 2. Best Non-Tree-Based Pipeline (Logistic Regression vs. KNN)
# Logistic Regression was selected because it achieved the highest
# performance among the non-tree-based classifiers tested.
# In Task 3, the scaled features achieved higher accuracy than the
# PCA-reduced features, so this production pipeline includes
# StandardScaler but does not include PCA.
non_tree_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(C=1.0, max_iter=1000, solver='liblinear'))
])

print("Fitting Best Non-Tree-Based Pipeline...")
non_tree_pipeline.fit(X_train, y_train)
non_tree_preds = non_tree_pipeline.predict(X_test)

print("\n[Pipeline Results]: Logistic Regression")
print(classification_report(y_test, non_tree_preds))

print("""
Pipeline Comparison Discussion:

The Random Forest pipeline achieved a higher accuracy (0.94)
compared with the Logistic Regression pipeline (0.93). This matches
the earlier results from Task 3 and Task 4, where Random Forest was
the strongest-performing model.

The two pipelines do not have the same structure because the models
have different requirements. The Random Forest pipeline only contains
the classifier because tree-based models do not require feature scaling.

The Logistic Regression pipeline includes StandardScaler before the
classifier because Logistic Regression is sensitive to feature scales.
Scaling helps improve model performance by putting all features on a
similar range.

The pipeline results are very close to the earlier manual approach,
which confirms that the preprocessing steps were applied correctly.
Using pipelines also helps prevent data leakage because preprocessing
is performed only using the training data.
""")




