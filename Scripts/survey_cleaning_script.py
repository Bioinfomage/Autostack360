
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import missingno as msno

# Setup for better visuals
pd.set_option('display.max_columns', None)

# Load data
df = pd.read_csv("C:/Users/kanmani/Desktop/AutoStack360/Data/Raw/2024_survey_results_public.csv")

# Shape and dimensions
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
df.info()
print(df.dtypes)
print(df.head())

# Summary statistics
print(df.describe())

# Total missing per column
missing = df.isnull().sum()
missing_counts = missing[missing > 0].sort_values(ascending=False)
print(missing_counts)

# Visualize missing data
msno.matrix(df, figsize=(24, 10), color=(0.3, 0.5, 0.8))
plt.title("Missing Value Matrix")
plt.show()

# Plot top 20 columns with most missing values
plt.figure(figsize=(10, 8))
sns.barplot(x=missing_counts[:20].values, y=missing_counts[:20].index, palette="viridis")
plt.title("Top 20 Columns with Missing Values")
plt.xlabel("Missing Value Count")
plt.ylabel("Column Name")
plt.tight_layout()
plt.show()

# Missing value percentage
missing_percent = (missing_counts / len(df)) * 100
missing_percent = missing_percent.round(2)
print(missing_percent.head(40))  # Show top 40

#Drop the rows that is not core developer data
df = df[df['MainBranch'] != 'I am not primarily a developer, but I write code sometimes as part of my work/studies']

# Drop irrelevant columns
explicit_drop = [
    "AINextMore integrated", "AINextMuch more integrated", "AINextMuch less integrated",
    "AINextLess integrated", "AINextNo change", "Frustration", "TimeSearching",
    "TimeAnswering", "ProfessionalTech", "ProfessionalCloud", "EmbeddedAdmired",
    "EmbeddedWantToWorkWith", "EmbeddedHaveWorkedWith", "Check", "Currency",
    "CompTotal", "SurveyLength", "SurveyEase", "WorkExp", "NEWSOSites"
]

pattern_drop = [col for col in df.columns if col.startswith("Knowledge_") or col.startswith("Frequency_")]
all_to_drop = explicit_drop + pattern_drop
df_cleaned = df.drop(columns=all_to_drop, errors="ignore")

# Confirm result
print(f"✅ Dropped {len(all_to_drop)} columns.")
print(f"📊 Remaining columns: {df_cleaned.shape[1]}")

# Save cleaned file
df_cleaned.to_csv(r"C:/Users/kanmani/Desktop/AutoStack360/Data/Cleaned/cleaned_survey_data.csv", index=False)

# Univariate Analysis
sns.set(style="whitegrid")

plt.figure(figsize=(10,5))
sns.histplot(df_cleaned['ConvertedCompYearly'], bins=50, kde=True)
plt.title("Salary Distribution")
plt.xlabel("Annual Salary (USD)")
plt.show()

# Log transform salary
df_cleaned['LogSalary'] = np.log1p(df_cleaned['ConvertedCompYearly'])

# Remove outliers
q99 = df_cleaned['LogSalary'].quantile(0.99)
df_cleaned = df_cleaned[df_cleaned['LogSalary'] < q99]
q05 = df_cleaned['LogSalary'].quantile(0.05)
df_cleaned = df_cleaned[df_cleaned['LogSalary'] > q05]

# Plot log salary
plt.figure(figsize=(10,5))
sns.histplot(df_cleaned['LogSalary'], bins=50, kde=True)
plt.title("Log-Transformed Salary Distribution")
plt.xlabel("Log(1 + Salary)")
plt.show()

# Drop original salary column and save
df_cleaned.drop(columns=['ConvertedCompYearly'], inplace=True)
output_path = r"C:/Users/kanmani/Desktop/AutoStack360/Data/Cleaned/log_transformed_survey_data.csv"
df_cleaned.to_csv(output_path, index=False)

print("✅ Cleaned data saved successfully!")
