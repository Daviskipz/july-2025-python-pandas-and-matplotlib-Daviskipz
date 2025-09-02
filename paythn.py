# -------------------------------
# Analyzing Data with Pandas and Visualizing Results with Matplotlib
# Using the Iris Dataset
# -------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# ===============================
# Task 1: Load and Explore the Dataset
# ===============================

try:
    # Load the Iris dataset from sklearn
    iris_data = load_iris(as_frame=True)
    df = iris_data.frame
    df['species'] = df['target'].map(dict(enumerate(iris_data.target_names)))  # Add species names
    
    # Display the first few rows
    print("First 5 rows of dataset:")
    print(df.head(), "\n")

    # Explore structure (data types & missing values)
    print("Dataset Info:")
    print(df.info(), "\n")

    print("Missing Values per Column:")
    print(df.isnull().sum(), "\n")

    # Clean dataset (no missing values in Iris, but just in case)
    df = df.dropna()

except FileNotFoundError:
    print("Error: Dataset file not found.")
except Exception as e:
    print("An error occurred:", str(e))

# ===============================
# Task 2: Basic Data Analysis
# ===============================

# Compute basic statistics
print("Basic Statistics of Numerical Columns:")
print(df.describe(), "\n")

# Perform grouping by categorical column (species)
grouped = df.groupby("species").mean()
print("Mean of numerical columns grouped by species:")
print(grouped, "\n")

# Example interesting finding
print("Observation: Iris-virginica generally has the highest mean petal length and width.\n")

# ===============================
# Task 3: Data Visualization
# ===============================

# Set seaborn style
sns.set(style="whitegrid")

# 1. Line Chart (trend of sepal length over first 50 samples as a time series)
plt.figure(figsize=(8, 5))
plt.plot(df.index[:50], df["sepal length (cm)"][:50], marker="o", label="Sepal Length")
plt.title("Line Chart: Sepal Length Trend (First 50 Samples)")
plt.xlabel("Sample Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.show()

# 2. Bar Chart (average petal length per species)
plt.figure(figsize=(8, 5))
sns.barplot(x="species", y="petal length (cm)", data=df, ci=None)
plt.title("Bar Chart: Average Petal Length by Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length (cm)")
plt.show()

# 3. Histogram (distribution of sepal width)
plt.figure(figsize=(8, 5))
plt.hist(df["sepal width (cm)"], bins=15, color="skyblue", edgecolor="black")
plt.title("Histogram: Distribution of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter Plot (sepal length vs petal length by species)
plt.figure(figsize=(8, 5))
sns.scatterplot(x="sepal length (cm)", y="petal length (cm)", hue="species", data=df)
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.show()

