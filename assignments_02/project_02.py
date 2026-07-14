# ----Part 2: Mini-Project -- Predicting Student Math Performance ----

# Observation: 
# In the raw student_performance_math.csv file, fields are separated 
# by semicolons (;) instead of standard commas (,). Categorical text values are enclosed 
# in double quotes (e.g., "yes", "no", "F", "M"), while numeric fields like age, absences, 
# G1, G2, and G3 are unquoted integers. To load this correctly with pd.read_csv(), 
# So, we must specify the `sep=";"` parameter beyond the filename.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Ensure outputs directory exists
os.makedirs("outputs", exist_ok=True)


# TASK 1: LOAD AND EXPLORE

print("--- TASK 1: LOAD AND EXPLORE ---")

# Load local dataset using the semicolon separator parameter
df = pd.read_csv("student_performance_math.csv", sep=";")

print(f"Dataset Shape: {df.shape}")

print("\nFirst 5 Rows:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

# Plot a histogram of G3 with 21 bins (0 to 20)
plt.figure(figsize=(8, 5))
plt.hist(df["G3"], bins=21, range=(0, 20), edgecolor="black", color="orange")
plt.title("Distribution of Final Math Grades")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Number of Students")
plt.grid(axis='y', alpha=0.75)

# Save the plot
plt.savefig("outputs/g3_distribution.png")
plt.close()
# print this as areminder
print("\nHistogram saved to 'outputs/g3_distribution.png'")


# TASK 2: PREPROCESS THE DATA

print("\n--- TASK 2: PREPROCESS THE DATA ---")
shape_before = df.shape
print(f"Shape before filtering zeros: {shape_before}")

# Filter out G3 = 0 rows to create a cleaned DataFrame
df_clean = df[df["G3"] > 0].copy()
shape_after = df_clean.shape
print(f"Shape after filtering zeros:  {shape_after}")
print(f"Rows removed: {shape_before[0] - shape_after[0]}")

# COMMENT: Why keeping G3=0 distorts the model
# A final grade of 0 usually indicates the student missed the final exam, dropped out, 
# or experienced an administrative issue rather than students who simply performed poorly.
# For example, a student might get a 0 because they missed the final exam or dropped out, not because they performed poorly.
# The model doesn't know that, so it tries to explain the 0 using features like study time or previous grades.
# This can confuse the model and make it harder to learn the real relationship between the input features and the final grade.


# Convert binary categorical columns (yes/no) to 1/0
binary_cols = ["schoolsup", "internet", "higher", "activities"]
for col in binary_cols:
    df_clean[col] = df_clean[col].map({"yes": 1, "no": 0})

# Convert sex column exactly as requested: F=0, M=1
df_clean["sex"] = df_clean["sex"].map({"F": 0, "M": 1})

# Compute Pearson correlation between absences and G3
corr_original = df["absences"].corr(df["G3"])
corr_filtered = df_clean["absences"].corr(df_clean["G3"])

print(f"\nPearson correlation between absences and G3 (Original Data): {corr_original:.4f}")
print(f"Pearson correlation between absences and G3 (Filtered Data): {corr_filtered:.4f}")

# COMMENT: Why filtering changes the correlation result
# In the original dataset, many students who received a G3 of 0 actually had low or zero absences. 
# Because they dropped out or missed the exam, they didn't accumulate class absences, yielding 
# data points close to (0 absences, 0 grade). This severely weakens the Pearson correlation, 
# making it look like absences don't matter. 
# Filtering changes the correlation because the data has changed. When we remove rows where G3 = 0, we're removing students who may have unusual or extreme values.
# Those rows can have a strong effect on the relationship between absences and G3.
# After removing them, the correlation is calculated using only the remaining students, so it may become stronger, weaker, or even change direction.


# TASK 3: EXPLORATORY DATA ANALYSIS

print("\n--- TASK 3: EXPLORATORY DATA ANALYSIS ---")

# Select numeric and newly converted binary columns
numeric_df = df_clean.select_dtypes(include=[np.number])

# Drop G1 and G2, then compute correlation with G3
correlations = numeric_df.drop(columns=["G1", "G2"]).corr()["G3"].sort_values()

print("Pearson correlation with G3 (Sorted from negative to positive):")
print(correlations)

# Visualizations
# Plot 1: Boxplot of G3 by number of past class failures
plt.figure(figsize=(8, 5))
sns.boxplot(x="failures", y="G3", data=df_clean, color="lightcoral")
plt.title("Final Grade (G3) Distribution by Past Failures")
plt.xlabel("Number of Past Class Failures")
plt.ylabel("Final Grade (G3)")
plt.savefig("outputs/failures_vs_g3.png")
plt.close()

# COMMENT (Plot 1): 
# This boxplot shows a clear downward trend in final grades as the number of past failures increases. 
# Students with 0 past failures have a median grade around 12, whereas students with 3 failures 
# have a significantly lower median grade and tighter distribution, highlighting 'failures' 
# as a powerful negative predictor.

# Plot 2: Scatter plot of Absences vs G3 with a trendline
plt.figure(figsize=(8, 5))
plt.scatter(df_clean["absences"], df_clean["G3"], alpha=0.6, color="green")
m, b = np.polyfit(df_clean["absences"], df_clean["G3"], 1)
plt.plot(df_clean["absences"], m*df_clean["absences"] + b, color="red", linestyle="--")
plt.title("Absences vs Final Grade (G3)")
plt.xlabel("Number of Absences")
plt.ylabel("Final Grade (G3)")
plt.savefig("outputs/absences_vs_g3.png")
plt.close()

# COMMENT (Plot 2): 
# The scatter plot demonstrates the negative relationship between absences and final grades. 
# The downward-sloping trendline confirms that higher absenteeism generally correlates with lower performance, 
# though there is significant variance among students with few absences.


# TASK 4: BASELINE MODEL

print("\n--- TASK 4: BASELINE MODEL ---")

# Define features (X) and target (y) using failures alone
X_baseline = df_clean[["failures"]]
y_baseline = df_clean["G3"]

# Split into training and test sets (80/20, random_state=42)
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_baseline, y_baseline, test_size=0.2, random_state=42)

# Initialize and fit the baseline model
baseline_model = LinearRegression()
baseline_model.fit(X_train_b, y_train_b)

# Make predictions on the test set
y_pred_b = baseline_model.predict(X_test_b)

# Calculate metrics
slope_b = baseline_model.coef_[0]
rmse_b = np.sqrt(mean_squared_error(y_test_b, y_pred_b))
r2_b = r2_score(y_test_b, y_pred_b)

print(f"Model Slope (Coefficient): {slope_b:.4f}")
print(f"Test RMSE: {rmse_b:.4f}")
print(f"Test R²: {r2_b:.4f}")

# COMMENT: Interpreting the metrics
# 1. Slope: The slope of roughly -1.16 means that, for every past class a student 
#    has failed, their predicted final math grade drops by about 1.16 points on the 0-20 scale.
# 2. RMSE: An RMSE of roughly 3.16 means that the baseline model's predictions are, on average, 
#    about 3.16 grade points off from the student's actual final grade. Given the 0-20 scale, this 
#    is a moderate error (around 15% of the total scale range).
# 3. R²: The R² score is roughly 0.07 (or 7%). This means that past failures alone explain only about 7% 
#    of the variance in final math grades. This low number matches expectations from the EDA, where the 
#    Pearson correlation for failures was around -0.29. Squaring that correlation gives roughly 0.08, 
#    confirming that while 'failures' is our strongest single predictor, background and behavioral data 
#    alone make it very difficult to predict academic performance accurately without earlier test grades (G1/G2).


# TASK 5: BUILD THE FULL MODEL

print("\n--- TASK 5: BUILD THE FULL MODEL ---")

# Explicit feature list configuration
feature_cols = ["failures", "Medu", "Fedu", "studytime", "higher", "schoolsup", "internet", "sex", "freetime", "activities", "traveltime"]

X = df_clean[feature_cols].values
y = df_clean["G3"].values

# Split into training and test sets (80/20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the full linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"Train R²: {train_r2:.4f}")
print(f"Test R²:  {test_r2:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")

print("\nFeatures and Coefficients:")
for name, coef in zip(feature_cols, model.coef_):
    print(f"{name:12s}: {coef:+.3f}")

# COMMENT: Surprising Coefficients and Train vs Test Gap
# 1. Surprising Sign: 'schoolsup' (school educational support) has a negative coefficient (~ -0.90). 
#    This seems counterintuitive because "support" should help grades. However, schoolsup = 1 means 
#    the student *receives remedial academic support*. Thus, this feature acts as a proxy indicator 
#    that a student was already struggling heavily, which explains why its coefficient is negative.
# 2. Train vs Test Gap: Train R² and Test R² are reasonably close (e.g., ~0.19 vs ~0.17). This indicates 
#    the model is not severely overfitting; it's simply limited by the weak linear relationships 
#    present in behavioral data alone.
# 
# Production Deployment Selection: 
# If deploying this model, I'd keep 'failures', 'Medu', 'higher', and 'schoolsup' because they possess 
# the strongest weight/coefficients and clear logical significance. I would drop 'activities', 'sex', and 
# 'freetime' because their coefficients are close to zero, meaning they add virtually no predictive 
# power and increase unnecessary data collection overhead.


# TASK 6: EVALUATE AND SUMMARIZE

print("\n--- TASK 6: EVALUATE AND SUMMARIZE ---")

# Create Predicted vs Actual Plot
plt.figure(figsize=(7, 7))
plt.scatter(y_pred_test, y_test, alpha=0.7, color="purple")

# Add a diagonal reference line
mn = min(min(y_pred_test), min(y_test))
mx = max(max(y_pred_test), max(y_test))
plt.plot([mn, mx], [mn, mx], color="red", linestyle="--", linewidth=2)

plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted Grade (y_hat)")
plt.ylabel("True Grade (y)")
plt.grid(True, alpha=0.3)
plt.savefig("outputs/predicted_vs_actual.png")
plt.close()

# COMMENT (Plot Interpretation):
# The model struggles significantly at the extremes. On the low end (actual grades below 8), 
# the model heavily overpredicts because it clusters predictions tightly between 10 and 13.
# A point ABOVE the diagonal means the actual grade was higher than predicted (underestimation). 
# A point BELOW the diagonal means the actual grade was lower than predicted (overestimation).

# SUMMARY COMMENTS:
# 1. The filtered dataset contains 357 rows (out of 395 originally), making the 20% test set 72 rows.
# 2. Performance: The best model achieves a test RMSE of ~2.9 points and an R² of ~0.17. 
#    a typical grade prediction is off by roughly 3 points on a 0-20 scale. It only accounts 
#    for 17% of why grades differ between students.
# 3. Key Drivers: 'higher' (wants to take higher education) has the largest positive coefficient, meaning 
#    high motivation yields higher scores. 'failures' has the largest negative coefficient, meaning past 
#    academic struggles strongly drag down final performance.
# 4. Surprise: Father's education ('Fedu') matters noticeably less than Mother's education ('Medu') 
#    according to the calculated coefficients.


# NEGLECTED FEATURE: THE POWER OF G1

print("\n--- NEGLECTED FEATURE: THE POWER OF G1 ---")

# Add G1 to our feature columns
feature_cols_with_g1 = feature_cols + ["G1"]

X_g1 = df_clean[feature_cols_with_g1].values
y_g1 = df_clean["G3"].values

# Split (80/20, random_state=42)
X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(X_g1, y_g1, test_size=0.2, random_state=42)

# Refit model
g1_model = LinearRegression()
g1_model.fit(X_train_g, y_train_g)

# Predict and print test R2
y_pred_test_g = g1_model.predict(X_test_g)
test_r2_g1 = r2_score(y_test_g, y_pred_test_g)

print(f"Test R² with G1 included: {test_r2_g1:.4f}")

# COMMENT (The Power of G1):
# 1. Does a high R² mean G1 causes G3? No, correlation is not causation. G1 does not "cause" G3; 
#    rather, G1 acts as an early, direct measurement of the *same underlying academic performance* 
#    and mastery of the subject matter that G3 tests later on.
# 2. Usefulness for Identification: Yes, it is highly useful for identifying students who are already 
#    struggling by the end of the first period, providing a clear statistical flag for intervention.
# 3. Early Intervention: If educators want to intervene *before* G1 data is even available, they 
#    cannot rely on this high-performing model. They would need to look back at our Task 5 behavioral 
#    model, prioritizing interventions based on student background indicators such as past class failures 
#    or whether the student lacks a desire to pursue higher education ('higher' = 0).

