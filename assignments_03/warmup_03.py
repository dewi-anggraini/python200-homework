
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# ---- Preprocessing ----

# Q1
# Split X and y into training and test sets using an 80/20 split with stratify=y and random_state=42.
# Print the shapes of all four arrays.
X_train, X_test, y_train, y_test = train_test_split( 
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
# Print the shapes
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

# Q2
# Fit a StandardScaler on X_train and use it to transform both X_train and X_test.

# Scale Using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Print the mean of each column in X_train_scaled -- they should all be very close to 0.
print("Means of scaled training columns:", X_train_scaled.mean(axis=0)) # calculates the mean down each column

# Add a comment explaining in one sentence why you fit the scaler on X_train only.
# COMMENT: why you fit the scaler on X_train only.
# Fit scaler only on training data to prevent data leakage from the test set.

# ---- KNN -----

# Q1
#Build a KNeighborsClassifier with n_neighbors=5, fit it on the unscaled training data (X_train), and predict on the test set.
# Print the accuracy score and the full classification report.

# Initialize the model with 5 neighbors
knn = KNeighborsClassifier(n_neighbors=5)

# Fit the model on UNSCALED training data
knn.fit(X_train, y_train)

# Predict on the test set
y_pred = knn.predict(X_test)

# Calculate and print accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy:.4f}\n")

# Print the full classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Q2
#Repeat KNN Question 1 using the scaled data (X_train_scaled, X_test_scaled). Print the accuracy score.
# Initialize the model with 5 neighbors
knn_scaled = KNeighborsClassifier(n_neighbors=5)

# Fit the model on SCALED training data
knn_scaled.fit(X_train_scaled, y_train)

# Predict on the scaled test set
y_pred_scaled = knn_scaled.predict(X_test_scaled)

# Calculate and print accuracy score
accuracy_scaled = accuracy_score(y_test, y_pred_scaled)
print(f"Scaled Data Accuracy Score: {accuracy_scaled:.4f}")

# Add a comment: does scaling improve performance, hurt it, or make no difference? Why might that be for this particular dataset?
# Comment: Scaling made no difference because Iris features have similar ranges and the dataset is easily separated.


# Q3
#Using cross_val_score with cv=5, evaluate the k=5 KNN model on the unscaled training data.
# Print each fold score, the mean, and the standard deviation.

# Initialize the model
knn_cv = KNeighborsClassifier(n_neighbors=5)

# Perform 5-fold cross-validation on the unscaled training data
cv_scores = cross_val_score(knn_cv, X_train, y_train, cv=5)

# Print each fold score, the mean, and the standard deviation
print("Individual Fold Scores:", cv_scores)
print(f"Mean CV Score        : {cv_scores.mean():.4f}")
print(f"Standard Deviation   : {cv_scores.std():.4f}")

# COMMENT: is this result more or less trustworthy than a single train/test split, and why?
# Cross-validation is more trustworthy than a single train/test split because it evaluates the model across multiple distinct subsets of the data, reducing the likelihood
# that the performance score is artificially inflated or deflated by a lucky or unlucky random split.

# Q4
#Loop over k values [1, 3, 5, 7, 9, 11, 13, 15]. For each,
# compute 5-fold cross-validation accuracy on the unscaled training data and print k and the mean CV score.

# List of k values to evaluate
k_values = [1, 3, 5, 7, 9, 11, 13, 15]

print("k value | Mean CV Score")
print("-----------------------")

# Loop over each k value
for k in k_values:
    knn_loop = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_loop, X_train, y_train, cv=5)
    print(f"k = {k:2d}   | {scores.mean():.4f}")

# COMMENT: identifying which k you would choose and why.
# I would choose k = 7 because it ties for the highest mean cross-validation accuracy (0.9750) while using a larger neighborhood than k = 3,
# which offers better smoothing against noise and reduces the risk of overfitting.

#---- Classifier Evaluation -----

# Q1

# Generate the confusion matrix using predictions from KNN Question 1
cm = confusion_matrix(y_test, y_pred)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap=plt.cm.Blues)

# Save the figure
plt.savefig("outputs/knn_confusion_matrix.png", bbox_inches='tight')
plt.show()

# COMMENT: which pair of species does the model most often confuse (if any)?
# For this particular dataset,
# the model achieved 100% accuracy on the test set, meaning it did not confuse any pair of species.

# --- The sklearn API: Decision Trees ---

# Q1

# Initialize the Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(max_depth=3, random_state=42)

# Fit on unscaled training data
dt_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred_dt = dt_classifier.predict(X_test)

# Print accuracy score
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f"Decision Tree Accuracy Score: {accuracy_dt:.4f}\n")

# Print the full classification report
print("Classification Report:")
print(classification_report(y_test, y_pred_dt))

# Comment comparing the Decision Tree accuracy to KNN.
# The Decision Tree accuracy (0.9667) is slightly lower than the unscaled KNN accuracy (1.0000), meaning it misclassified one sample that KNN predicted correctly.

# Second Comment: given that Decision Trees don't rely on distance calculations, would scaled vs. unscaled data affect the result?
# Scaling the data would not affect the Decision Tree result because tree-based models split features based on value thresholds rather than distance metrics,
# making them invariant to monotonic feature scaling.

# --- Logistic Regression and Regularization ---

# Q1
# One-vs-Rest wrapper enables multiclass logistic regression with liblinear.

for C in [0.01, 1.0, 100]:
    log_reg = OneVsRestClassifier(
        LogisticRegression(
            C=C,
            max_iter=1000,
            solver="liblinear",
        )
    )

    log_reg.fit(X_train_scaled, y_train)
    coef_sum = np.abs(
        np.vstack([est.coef_ for est in log_reg.estimators_])
    ).sum()
    # coef_sum = np.abs(log_reg.estimators_[0].coef_).sum()
    print(f"C={C}, total coefficient magnitude={coef_sum}")


# COMMENT: what happens to the total coefficient magnitude as C increases? # What does this tell you about what regularization is doing?
# As C increases, the coefficient magnitude increases because weaker regularization allows larger weights.
# Smaller C applies stronger regularization and keeps coefficients smaller.


# ---- PCA ---

# Data-loading block
digits = load_digits()
X_digits = digits.data    # 1797 images, each flattened to 64 pixel values
y_digits = digits.target  # digit labels 0-9
images = digits.images    # same data shaped as 8x8 images for plotting

# Q1

# Print the shapes
print("X_digits shape:", X_digits.shape)
print("images shape  :", images.shape)

# Create a 1-row subplot showing one example of each digit class (0-9)
fig, axes = plt.subplots(1, 10, figsize=(15, 3))

for digit in range(10):
    # Find the index of the first image corresponding to the current digit
    index = next(i for i, label in enumerate(y_digits) if label == digit)
    
    # Display the image using the reversed grayscale colormap
    axes[digit].imshow(images[index], cmap='gray_r')
    axes[digit].set_title(f"Label: {digit}")
    axes[digit].axis('off')  # Hide axis ticks for cleaner visual

plt.tight_layout()

# Save and show the figure
plt.savefig("outputs/sample_digits.png", bbox_inches='tight')
plt.show()

# Q2

# Fit PCA on X_digits (retains all 64 components by default)
pca = PCA()
pca.fit(X_digits)

# Get the projection scores
scores = pca.transform(X_digits)

# Create the scatter plot using the first two Principal Components
plt.figure(figsize=(8, 6))
scatter = plt.scatter(scores[:, 0], scores[:, 1], c=y_digits, cmap='tab10', s=10, alpha=0.7)

# Add colorbar, labels, and title
plt.colorbar(scatter, label='Digit')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D PCA Projection of Handwritten Digits')

# Save and show the figure
plt.savefig("outputs/pca_2d_projection.png", bbox_inches='tight')
plt.show()

# COMMENT: do same-digit images tend to cluster together in this 2D space?
# Yes, same-digit images visibly tend to cluster together in this 2D space,
# though there is some overlap between certain classes due to the structural complexity that cannot be fully captured in only two dimensions.

# Q3

# Calculate the cumulative sum of the explained variance ratio
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Plot cumulative explained variance vs. number of components
plt.figure(figsize=(8, 5))
# Components are 1-indexed for the plot (1 to 64)
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-', markersize=4)

# Add a horizontal line at 80% variance for reference
plt.axhline(y=0.80, color='r', linestyle='--', label='80% Explained Variance')

plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Cumulative Explained Variance vs. Number of Components')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()

# Save and show the figure
plt.savefig("outputs/pca_variance_explained.png", bbox_inches='tight')
plt.show()

# Comment: approximately how many components do you need to explain 80% of the variance?
# Based on the data,
# need approximately 13 components to explain 80% of the variance in the digits dataset.

# Q4

def reconstruct_digit(sample_idx, scores, pca, n_components):    
    """Reconstruct one digit using the first n_components principal components."""    
    reconstruction = pca.mean_.copy()    
    for i in range(n_components):        
        reconstruction = reconstruction + scores[sample_idx, i] * pca.components_[i]    
    return reconstruction.reshape(8, 8)

# Configuration for the subplot grid
n_values = [2, 5, 15, 40]
num_digits = 5

# Create a grid of (1 row for original + len(n_values) rows) by 5 columns
fig, axes = plt.subplots(len(n_values) + 1, num_digits, figsize=(10, 12))

# Row 0: Plot the original images
for col in range(num_digits):
    axes[0, col].imshow(images[col], cmap='gray_r')
    axes[0, col].axis('off')

axes[0, 0].set_ylabel(
    "Original",
    rotation=0,
    labelpad=60,
    verticalalignment='center',
    fontweight='bold'
)

# Rows 1 to 4: Plot reconstructions for each n_components value
for row_idx, n in enumerate(n_values, start=1):
    for col in range(num_digits):
        reconstructed_img = reconstruct_digit(col, scores, pca, n)
        axes[row_idx, col].imshow(reconstructed_img, cmap='gray_r')
        axes[row_idx, col].axis('off')

    # Label the row with the number of components used
    axes[row_idx, 0].set_ylabel(
        f"n = {n} components",
        rotation=0,
        labelpad=60,
        verticalalignment='center',
        fontweight='bold'
    )

plt.tight_layout()

# Save and show the figure
plt.savefig("outputs/pca_reconstructions.png", bbox_inches='tight')
plt.show()

# Comment:
# Observation: the digits appear more recognizable around n = 15,
# and this is roughly consistent with the point where the explained variance curve begins to level off.
