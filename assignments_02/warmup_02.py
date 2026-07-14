# ---The scikit-learn API ---

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# scikit-learn Question 1
years  = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])
new_years = np.array([4, 8]).reshape(-1, 1)

model = LinearRegression()                    # 1. create model
model.fit(years, salary)                 # 2. fit model to data (learn)
predicted_salary = model.predict(new_years)            # 3. predict with new data
  
print("Predicted salary for 4 years:", predicted_salary[0])
print("Predicted salary for 8 years:", predicted_salary[1])

print("Slope:", model.coef_[0]) 
print("Intercept:", model.intercept_)


# scikit-learn Question 2
# scikit-learn requires the feature array X to be 2D even when you only have one feature.
# Start with this 1D array:
x = np.array([10, 20, 30, 40, 50])
print("Original Shape:", x.shape)

x = x.reshape(-1, 1)
print("New Shape:", x.shape)

# Comment: 
# scikit-learn expects X to be 2D because each row is one sample
# and each column represents one feature.
# For example: If I only have one feature, numpy sees this as shape (2,).
# Scikit-learn can't tell whether this means: 2 samples with 1 feature, or 1 sample with 2 features.
# To remove this ambiguity, scikit-learn requires X to be 2D.

# scikit-learn Question 3
# K-Means is an unsupervised algorithm that follows the same create → fit → predict pattern as everything else in scikit-learn.
# Use the code below to generate a synthetic dataset with three natural clusters:

X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)
print("Dataset Shape", X_clusters.shape)

kmeans = KMeans(n_clusters=3, random_state=7)  # 1. Create the model
kmeans.fit(X_clusters)                                    # 2. Fit -- find cluster centers
labels = kmeans.predict(X_clusters)                       # 3. Predict a label for each point

# Print the cluster centers
print("\nCluster Centers:")
print(kmeans.cluster_centers_)

# Print how many points are in each cluster
print("\nPoints in each cluster:")
print(np.bincount(labels))

# Create the scatter plot
plt.figure(figsize=(8, 6))

# Plot all points, colored by cluster
plt.scatter(
    X_clusters[:, 0],
    X_clusters[:, 1],
    c=labels,
    cmap="viridis",
    s=60,
    alpha=0.8
)

# Plot the cluster centers as black X's
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    marker="X",
    color="black",
    s=200,
    label="Cluster Centers"
)

# Add title and axis labels
plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()

# Create the outputs folder if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# Save the figure
plt.savefig("outputs/kmeans_clusters.png")

# Display the figure
plt.show()

# --- Linear Regression ---
# Linear Regression Question 1
# Before fitting anything, look at the data. Create a scatter plot of age on the x-axis and cost on the y-axis.
# Color the points by smoker status by passing c=smoker and cmap="coolwarm" to plt.scatter(). Add a title "Medical Cost vs Age", label both axes, and save to outputs/cost_vs_age.png.
# Add a comment describing what you see. Are there two distinct groups visible? What does that suggest about the smoker variable?
np.random.seed(42)
num_patients = 100
age    = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost   = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)

plt.figure(figsize=(8, 6))

plt.scatter(
    age,
    cost,
    c=smoker,
    cmap="coolwarm",
    alpha=0.7
)

plt.title("Medical Cost vs Age")
plt.xlabel("Age")
plt.ylabel("Annual Medical Cost")

os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/cost_vs_age.png")

plt.show()

# Comments:
# The points appear to form two separate groups.
# Smokers generally have much higher medical costs.
# This suggests smoking is an important predictor of cost.


# Linear Regression Question 2
# Split the data into training and test sets using age as the only feature, an 80/20 split, and random_state=42.
# Reshape age to a 2D array before using it as X. Print the shapes of all four arrays.

# Reshape age so it can be used as X
X = age.reshape(-1, 1)
y = cost

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Print the shapes
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


# Linear Regression Question 3
# Fit a LinearRegression model to your training data from Question 2. Print the slope and intercept.
# Then predict on the test set and print:
# Create the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Print model parameters
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))

# Calculate R²
r2 = model.score(X_test, y_test)

print("RMSE:", rmse)
print("R²:", r2)

# Interpretation:
# The slope tells how much the predicted medical cost changes for each additional year of age.
# For example, if the slope is about 200, then every extra year
# of age increases the predicted annual medical cost by about $200.

# Linear Regression Question 4
# Now add smoker as a second feature and fit a new model.
# Create a feature matrix using both age and smoker
X_full = np.column_stack([age, smoker])

# Split the data
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_full,
    cost,
    test_size=0.2,
    random_state=42
)

# Create and train the model
model_full = LinearRegression()
model_full.fit(X_train_f, y_train_f)

# Calculate R²
r2_full = model_full.score(X_test_f, y_test_f)

print("R² using only age:", r2)
print("R² using age + smoker:", r2_full)

print("Age coefficient:", model_full.coef_[0])
print("Smoker coefficient:", model_full.coef_[1])

# Interpretation:
# The smoker coefficient tells how much more (or less) the predicted annual medical cost changes for smokers
# compared with non-smokers, after accounting for age.
# Example:
# if the coefficient is around 15000,
# smokers are predicted to cost about $15,000 more per year than non-smokers of the same age.
# a smoker is predicted to have medical costs higher than a non-smoker.

# Linear Regression Question 5
# A predicted vs actual plot is a standard tool for evaluating regression models. Each test observation becomes a dot: the model's prediction goes on the x-axis, the true value goes on the y-axis. A perfect model would place every point on the diagonal line where predicted equals actual.
#Using the two-feature model from Linear Regression Question 4, create this plot for the test set. Add a diagonal reference line, a title "Predicted vs Actual", labeled axes, and save to outputs/predicted_vs_actual.png.
# Add a comment: what does it mean when a point falls above the diagonal? What about below?

# Predict on the test set
y_pred = model_full.predict(X_test_f)

plt.figure(figsize=(7, 7))

# Scatter plot
plt.scatter(y_pred, y_test_f, alpha=0.7)

# Diagonal reference line
minimum = min(y_pred.min(), y_test_f.min())
maximum = max(y_pred.max(), y_test_f.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    color="red",
    linestyle="--",
    label="Perfect Prediction"
)

plt.title("Predicted vs Actual")
plt.xlabel("Predicted Cost")
plt.ylabel("Actual Cost")
plt.legend()

os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/predicted_vs_actual.png")

plt.show()

# Interpretation:
# Points above the diagonal have actual costs that are higher
# than the model predicted (the model underestimated the cost).
#
# Points below the diagonal have actual costs that are lower
# than the model predicted (the model overestimated the cost).