import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create output folder if it doesn't exist
os.makedirs("output", exist_ok=True)

# Load dataset
df = pd.read_csv("titanic.csv")

# Display first 5 rows
print("\nFirst 5 Rows:")
print(df.head())

# Display dataset information
print("\nDataset Information:")
print(df.info())

# Display shape of dataset
print("\nShape of Dataset:")
print(df.shape)

# Display column names
print("\nColumn Names:")
print(df.columns)

# Statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate records
print("\nDuplicate Records:")
print(df.duplicated().sum())

# Remove duplicate records
df.drop_duplicates(inplace=True)

# -------------------------------
# Histogram: Age Distribution
# -------------------------------
plt.figure(figsize=(8, 5))
plt.hist(df['Age'].dropna(), bins=20)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Age Distribution")
plt.savefig("output/age_distribution.png")
plt.close()

# -------------------------------
# Gender Count Plot
# -------------------------------
plt.figure(figsize=(6, 5))
sns.countplot(x='Sex', data=df)
plt.title("Gender Distribution")
plt.savefig("output/gender_distribution.png")
plt.close()

# -------------------------------
# Scatter Plot: Age vs Fare
# -------------------------------
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Age', y='Fare', data=df)
plt.title("Age vs Fare")
plt.savefig("output/age_vs_fare.png")
plt.close()

# -------------------------------
# Box Plot: Survival vs Age
# -------------------------------
plt.figure(figsize=(8, 5))
sns.boxplot(x='Survived', y='Age', data=df)
plt.title("Survival vs Age")
plt.savefig("output/survival_vs_age.png")
plt.close()

# -------------------------------
# Correlation Heatmap
# -------------------------------
plt.figure(figsize=(8, 6))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.savefig("output/correlation_heatmap.png")
plt.close()

# -------------------------------
# Survival Count Plot
# -------------------------------
plt.figure(figsize=(6, 5))
sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.savefig("output/survival_count.png")
plt.close()

print("\nEDA completed successfully!")
print("Graphs have been saved in the 'output' folder.")