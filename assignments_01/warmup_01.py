# --- Pandas ---
# Pandas Q1
# Create the following DataFrame and print the first three rows, the shape, and the data types of each column.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics
from scipy import stats
from scipy.stats import pearsonr
import seaborn as sns

data = {
    "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade":  [85, 72, 90, 68, 95],
    "city":   ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True]
}
df = pd.DataFrame(data)

print("First 3 rows:") # sometimes I still feel a little confused when I use the f-strings
print(df.head(3))

print(f"Shape: {df.shape}")

print("Data Types:")
print(df.dtypes)

# Pandas Q2.
# Using the DataFrame from Q1, filter the rows to show only students who passed and have a grade above 80. Print the result.

# step1: Filter rows: grade > 80 AND passed == True
df_filtered = df[(df["grade"] > 80) & (df["passed"] == True)]

# step2: print the result.
print(f"Students who passed and scored aboove 80:\n{df_filtered}")

#Pandas Q3
# Add a new column called "grade_curved" that adds 5 points to each student's grade.
# Print the updated DataFrame (all columns, all rows).

df["grade_curved"]= df["grade"] + 5  # my first try
# df = df.assign(grade_curved=df["grade"] + 5) # try .assign()
print(df)

# Pandas Q4
# Add a new column called "name_upper" that contains each student's name in uppercase, using the .str accessor.
# Print the "name" and "name_upper" columns together.

df["name_upper"] = df["name"].str.upper()
print(df[["name", "name_upper"]])

# Pandas Q5
# Group the DataFrame by "city" and compute the mean grade for each city. Print the result.
#df = df.groupby("city").mean(), I was computing all the numeric columns (grade, grade_curved, etc) so, I put the city means in a new variable

city_means = df.groupby("city")["grade"].mean()
print(city_means)

# Pandas Q6
# Replace the value "Austin" in the "city" column with "Houston".
# choose the column, which is "city", then replace the value
df["city"] = df["city"].replace("Austin", "Houston")

# Print the "name" and "city" columns to confirm the change.
print(df[["city", "name"]])

# Pandas Q7
# Sort the DataFrame by "grade" in descending order and print the top 3 rows.

# df['grade'].desc, since pandas does not have .desc attribute
df_sorted = df.sort_values("grade", ascending=False)
print(df_sorted.head(3))

# ----NumPy---
# NumPy Q1
# Create a 1D NumPy array from the list [10, 20, 30, 40, 50]. Print its shape, dtype, and ndim.
arr = np.array([10, 20, 30, 40, 50]) # a np.array([]) is a function, always use () outside []
print(arr)
print(f"Shape: {arr.shape}")
print(f"Dtype : {arr.dtype}")
print(f"Dimension: {arr.ndim}")

# NumPy Q2
# Create the following 2D array and print its shape and size (total number of elements).
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
print(f"Shape: {arr.shape}")
print(f"Size: {arr.size}")

# NumPy Q3
# Using the 2D array from Q2, slice out the top-left 2x2 block and print it.
# The expected result is [[1, 2], [4, 5]].
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
# using block (2x2 block)
block = arr[0:2, 0:2]
print(block)

# NumPy Q4
# Create a 3x4 array of zeros using a built-in command. 
# Then create a 2x5 array of ones using a built-in command.
# To make 2D arrays, first, pass a tuple of dimensions, like (3, 4) or (2, 5).
arr_zeros = np.zeros((3, 4)) # built-in command
arr_ones = np.ones((2, 5))
# Print both.
print(arr_zeros) 
print(arr_ones)

# NumPy Q5
# Create an array using np.arange(0, 50, 5). First, think about what you expect it to look like.
# Then, print the array, its shape, mean, sum, and standard deviation.
arr = np.arange(0, 50, 5) 
print("Array:", arr)
print("Shape :", arr.shape)
print("Mean (average):", arr.mean()) 
print("Sum:", arr.sum()) 
print("Standard Deviation:", arr.std()) 

# NumPy Q6
# Generate an array of 200 random values drawn from a normal distribution with mean 0 and standard deviation 1 (use np.random.normal()).
# Print the mean and standard deviation of the result.
arr = np.random.normal(loc=0, scale=1, size=200)
print("mean:", arr.mean())
print("deviation:", arr.std())

# ---matplotlib---

# Matplotlib Q1
# Plot the following data as a line plot. Add a title "Squares", x-axis label "x", and y-axis label "y".
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]
plt.plot(x, y)
plt.title("Squares")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Matplotlib Q2
# Create a bar plot for the following subject scores. Add a title "Subject Scores" and label both axes.
subjects = ["Math", "Science", "English", "History"]
scores   = [88, 92, 75, 83]
plt.bar(subjects, scores, color="blue")
plt.title("Subject Scores")
plt.xlabel("Subjects")
plt.ylabel("Scores")
plt.show()

# Matplotlib Q3
# Plot the two datasets below as a scatter plot on the same figure.
# Use different colors for each, add a legend, and label both axes.
x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]
plt.scatter(x1, y1, color="green", label="Dataset 1")
plt.scatter(x2, y2, color="orange", label="Dataset 2")
plt.title("A Scatter Plot")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

# Matplotlib Q4
# Use plt.subplots() to create a figure with 1 row and 2 subplots side by side.
# In the left subplot, plot x vs y from Q1 as a line. In the right subplot, plot the subjects and scores from Q2 as a bar plot.
# Add a title to each subplot and call plt.tight_layout() before showing.
# Q1 data (Squares)
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]

# Q2 data (Subject Scores)
subjects = ["Math", "Science", "English", "History"]
scores   = [88, 92, 75, 83]

# Subplot = one figure (the frame) that contains two separate plots (Q1 and Q2) side by side.
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Left plot
axes[0].plot(x, y, color="blue")
axes[0].set_title("Squares")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")

# Right plot
axes[1].bar(subjects, scores, color="orange")
axes[1].set_title("Subject Scores")
axes[1].set_xlabel("Subjects")
axes[1].set_ylabel("Scores")

plt.tight_layout() # djusts the spacing between subplots, avoids overlap.
plt.show()

# ---Descriptive Statistic Review---

# Descriptive Stats Q1
# Given the list below, use NumPy to compute and print the mean, median, variance, and standard deviation. Label each printed value.

data = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]
arr = np.array(data) 
print("Mean (average):", arr.mean()) 
print("Median:", np.median(arr))
print("Variance:", arr.var()) 
print("Standard Deviation:", arr.std())

# Descriptive Stats Q2
# Generate 500 random values from a normal distribution with mean 65 and standard deviation 10 (use np.random.normal(65, 10, 500)).
# Plot a histogram with 20 bins. Add a title "Distribution of Scores" and label both axes.
data = np.random.normal(65, 10, 500)

plt.hist(data, bins=20, color="orange", edgecolor="black")
plt.title("Distribution of Scores")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.show()

# Descriptive Stats Q3
# Create a boxplot comparing the two groups below.
# Label each box ("Group A" and "Group B") and add a title "Score Comparison".
group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]

plt.boxplot([group_a, group_b], tick_labels=["Group A", "Group B"])
plt.title("Score Comparison")
plt.ylabel("Scores")
plt.show()

# Descriptive Stats Q4
# You are given two datasets: one normally distributed and one 'exponential' distribution.
# Create side-by-side boxplots comparing the two distributions.
# Label each boxplot appropriately ("Normal" and "Exponential") and add a title "Distribution Comparison".
# Then, add a comment in your code briefly noting which distribution is more skewed,
# and which descriptive statistic (mean or median) would provide a more appropriate measure of central tendency for each distribution.
normal_data = np.random.normal(50, 5, 200) # mean=50, std=5
skewed_data = np.random.exponential(10, 200) # exponential distribution

plt.boxplot([normal_data, skewed_data], tick_labels=["Normal", "Exponential"])
plt.title("Distribution Comparison")
plt.ylabel("Values")
plt.show()

# Comment:
# the exponential distribution is more skewed (long tail to the right)
# for the normal distribution, the mean is a good measure of central tendency
# for the skewed exponential distribution, the median is more appropriate.

# Descriptive Stats Q5
# Print the mean, median, and mode of the following:
# Why are the median and mean so different for data2? Add your answer as a comment in the code.

data1 = [10, 12, 12, 16, 18]
data2 = [10, 12, 12, 16, 150]

# convert data to NumPy arrays
arr1 = np.array(data1)
arr2 = np.array(data2)

print("Data1 Mean:", arr1.mean())
print("Data1 Median:", np.median(arr1))
print("Data1 Mode:", statistics.mode(arr1))
#print("Data1 Mode:", statistics.mode(arr1, keepdims=True).mode[0])

print("Data2 Mean:", arr2.mean())
print("Data2 Median:", np.median(arr2))
print("Data2 Mode:", statistics.mode(arr2))
# print("Data2 Mode:", statistics.mode(arr2, keepdims=True).mode[0])

# comment:
# Data1: No extreme values > mean = median, both are good measures.
# Data2: extreme value = 150, The outlier 150 drags the mean way up to 40,
# but the median (14) still reflects the “typical” middle value.

# ---- Hypothesis Testing Review----
# Hypothesis Question 1
# Run an independent samples t-test on the two groups below. Print the t-statistic and p-value.

group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]

# Perform independent t-test
t_stat, p_value_q1 = stats.ttest_ind(group_a, group_b)

print("t-statistic:", t_stat)
print("p-value:", p_value_q1)

# Hypothesis Question 2
# Using the p-value from Q1,
# write an if/else statement that prints whether the result is statistically significant at alpha = 0.05.
alpha = 0.05
if p_value_q1 < alpha:
    print("Result is statistically significant at alpha = 0.05 ")
else:
    print("Result is NOT statistically significant at alpha = 0.05") 

# Hypothesis Question 3
# Run a paired t-test on the before/after scores below (the same students measured twice). Print the t-statistic and p-value.

before = [60, 65, 70, 58, 62, 67, 63, 66]
after  = [68, 70, 76, 65, 69, 72, 70, 71]
# Perform independent t-test
t_stat, p_value = stats.ttest_rel(before, after)

print("t-statistic:", t_stat)
print("p-value:", p_value)

# Hypothesis Question 4
# Run a one-sample t-test to check whether the mean of scores is significantly different from a national benchmark of 70. Print the t-statistic and p-value.
scores = [72, 68, 75, 70, 69, 74, 71, 73]
t_stat, p_value = stats.ttest_1samp(scores, 70) # benchmark = 70

print("t-statistic:", t_stat)
print("p-value:", p_value)

# Hypothesis Question 5
# Re-run the test from Q1 as a one-tailed test to check whether group_a scores are less than group_b scores.
# Print the resulting p-value. Use the alternative parameter.

# To check specifically if group_a < group_b.
group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]

# Perform independent t-test
t_stat, p_value = stats.ttest_ind(group_a, group_b, alternative="less") #alternative parameter

print("one-tailed p-value:", p_value)

# Hypothesis Question 6
# Write a plain-language conclusion for the result of Q1 (do not just say "reject the null hypothesis"). Format it as a print() statement.
# Your conclusion should mention the direction of the difference and whether it is likely due to chance.
if p_value_q1 < 0.05:
    print("Group A scored lower than Group B, and this difference is unlikely due to chance.")
else:
    print("Group A and Group B scores are not different enough to rule out chance.")

# ---Correlation Review---
# Correlation Q1
# Compute the Pearson correlation between x and y below using np.corrcoef().
# Print the full correlation matrix, then print just the correlation coefficient (the value at position [0, 1]).
# What do you expect the correlation to be, and why? Add your answer as a comment in the code.
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10] # the y is 2*x = 2*1=2, 2*2=4,.....
# Compute correlation matrix
corr = np.corrcoef(x, y)

# Print full matrix
print("Correlation matrix:\n", corr)

# Print just the correlation coefficient
print("Correlation coefficient:", corr[0, 1])

# Expectation: Since y = 2 * x, the relationship is perfectly linear.
# So the correlation should be very close to 1.0 (perfect positive correlation).
# Correlation coefficient = 1.0. Because y is exactly double x, they have a perfect positive linear relationship.

# Correlation Q2
# Use pearsonr() from scipy.stats to compute the correlation between x and y below.
# Print both the correlation coefficient and the p-value.

x = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
y = [10, 9,  7,  8,  6,  5,  3,  4,  2,  1]

corr_coef, p_value = pearsonr(x, y)

print("Correlation coefficient:", corr_coef)
print("p-value:", p_value)

# Correlation Q3
# Create the following DataFrame and use df.corr() to compute the correlation matrix. Print the result.
people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55,  60,  65,  72,  80],
    "age":    [25,  30,  22,  35,  28]
}
df = pd.DataFrame(people)
print(df.corr())

# Correlation Q4
# Create a scatter plot of x and y below, which have a negative relationship.
# Add a title "Negative Correlation" and label both axes.
x = [10, 20, 30, 40, 50]
y = [90, 75, 60, 45, 30]

plt.scatter(x, y)
plt.title("Negative Correlation")
plt.xlabel("X values")
plt.ylabel("Y values")
plt.show()

# Correlation Q5
# Using the correlation matrix from Q3, create a heatmap with sns.heatmap().
# Pass annot=True so the correlation values appear in each cell, and add a title "Correlation Heatmap".
corr_matrix = df.corr()

sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ---Pipeline Review ---
# Pipeline Q1
# Step 1: Create Series
def create_series(arr):
    return pd.Series(arr, name="values")

# Step 2: Clean Data
def clean_data(series):
    return series.dropna()

# Step 3: Summarize Data
def summarize_data(series):
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]  # mode() returns a Series, and take first value
    }

# Step 4: Pipeline
def data_pipeline(arr):
    series = create_series(arr)
    cleaned = clean_data(series)
    summary = summarize_data(cleaned)
    return summary

# Given array
arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

# Run pipeline
result = data_pipeline(arr)

# Print results
for key, value in result.items():
    print(f"{key}: {value}")
