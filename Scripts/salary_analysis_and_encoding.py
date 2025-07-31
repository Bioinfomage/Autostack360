
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Setup for better visuals
pd.set_option('display.max_columns', None)

# Update the path to your downloaded file
file_path = r"C:\Users\kanmani\Desktop\AutoStack360\Data\Cleaned\log_transformed_survey_data.csv"
df = pd.read_csv(file_path)

# Quick check
print("✅ Data loaded successfully!")
print(df.shape)
df.head()#Numerical_Features
print(df[['YearsCodePro', 'LogSalary']].dtypes)# Replace textual values
df['YearsCodePro'] = df['YearsCodePro'].replace({
    'Less than 1 year': 0,
    'More than 50 years': 51
})
# Convert both to numeric
df['YearsCodePro'] = pd.to_numeric(df['YearsCodePro'], errors='coerce')
#checking
print(df[['YearsCode', 'YearsCodePro']].head())
print(df[['YearsCodePro', 'LogSalary']].dtypes) 
#scatterPlot
sns.scatterplot(x='YearsCodePro', y='LogSalary', data=df)
plt.title("Years of Professional Experience vs LogSalary")
plt.show()# Set a more appealing style
sns.set(style="whitegrid")

# Create the scatter plot with regression line and color by country
plt.figure(figsize=(12, 6))
sns.scatterplot(x='YearsCodePro', y='LogSalary', hue='Country', data=df, alpha=0.6, edgecolor=None)
# Add a regression line WITHOUT hue (since hue disables regplot)
sns.regplot(x='YearsCodePro', y='LogSalary', data=df, scatter=False, color='black', line_kws={'label':'Regression Line'})
# Add plot title and labels
plt.title("Years of Professional Experience vs LogSalary", fontsize=14)
plt.xlabel("Years of Professional Experience", fontsize=12)
plt.ylabel("Log(Annual Salary)", fontsize=12)

# Add legend for regression line and countries
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.tight_layout()

plt.show()print("Number of unique countries:", df['Country'].nunique())# Count of each country
country_counts = df['Country'].value_counts()

# Print the result
print(country_counts)

#plotting the country and it counts only above 100
# Step 2: Filter countries with more than 100 responses
filtered_countries = country_counts[country_counts > 100]

# Step 3: Plot using Seaborn
plt.figure(figsize=(12, 8))
sns.barplot(x=filtered_countries.values, y=filtered_countries.index, palette='mako')
plt.title('Countries with More Than 100 Responses')
plt.xlabel('Number of Responses')
plt.ylabel('Country')
plt.tight_layout()
plt.show()#Categorical vs. LogSalary: Use Boxplots + Group Means
# Group-wise average LogSalary
print(df.groupby('Country')['LogSalary'].mean().sort_values(ascending=False))

plt.figure(figsize=(14,6))
sns.boxplot(x='Country', y='LogSalary', data=df)
plt.xticks(rotation=45)
plt.title("LogSalary by Country")
plt.show()plt.figure(figsize=(14,6))
sns.boxplot(x='Country', y='LogSalary', data=df)
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.title("LogSalary by Country")
plt.xlabel("Country")
plt.ylabel("Log of Salary")
plt.tight_layout()       # Optional: Prevent label overlap
plt.show()# Step 1: Get Top 10 Countries by Mean LogSalary
top_countries = df.groupby('Country')['LogSalary'].mean().sort_values(ascending=False).head(10).index

# Step 2: Filter DataFrame
top_df = df[df['Country'].isin(top_countries)]

# Step 3: Plot
plt.figure(figsize=(10,6))
sns.boxplot(x='Country', y='LogSalary', data=top_df)
plt.xticks(rotation=45)
plt.title("LogSalary by Top 10 Countries")
plt.xlabel("Country")
plt.ylabel("Log of Salary")
plt.tight_layout()
plt.show()# Step 1: Get Top 10 Countries by Mean LogSalary
bottom_countries = df.groupby('Country')['LogSalary'].mean().sort_values(ascending=False).tail(10).index

# Step 2: Filter DataFrame
bottom_df = df[df['Country'].isin(bottom_countries)]

# Step 3: Plot
plt.figure(figsize=(10,6))
sns.boxplot(x='Country', y='LogSalary', data=bottom_df)
plt.xticks(rotation=45)
plt.title("LogSalary by bottom 10 Countries")
plt.xlabel("Country")
plt.ylabel("Log of Salary")
plt.tight_layout()
plt.show()# Grouping
salary_means = df.groupby('Country')['LogSalary'].mean().sort_values()
bottom_countries = salary_means.head(10).index
top_countries = salary_means.tail(10).index

# Setup figure
fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

# Bottom 10
sns.boxplot(x='Country', y='LogSalary', data=df[df['Country'].isin(bottom_countries)], ax=axes[0])
axes[0].set_title("LogSalary by Bottom 10 Countries")
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=90)
axes[0].set_xlabel("Country")
axes[0].set_ylabel("Log of Salary")

# Top 10
sns.boxplot(x='Country', y='LogSalary', data=df[df['Country'].isin(top_countries)], ax=axes[1])
axes[1].set_title("LogSalary by Top 10 Countries")
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=90)
axes[1].set_xlabel("Country")
axes[1].set_ylabel("")

plt.tight_layout()
plt.show()from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

def leakage_free_target_encoding(df, categorical_col, target_col, n_splits=5, random_state=42):
    df_copy = df.copy()
    encoded_col = f"{categorical_col}_Encoded"
    df_copy[encoded_col] = np.nan

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for train_idx, val_idx in kf.split(df_copy):
        train_df, val_df = df_copy.iloc[train_idx], df_copy.iloc[val_idx]
        target_mean = train_df.groupby(categorical_col)[target_col].mean()
        df_copy.loc[val_idx, encoded_col] = val_df[categorical_col].map(target_mean)

    global_mean = df[target_col].mean()
    df_copy[encoded_col] = df_copy[encoded_col].fillna(global_mean)
    return df_copy

df = leakage_free_target_encoding(df, categorical_col='Country', target_col='LogSalary')
df.drop(columns=['Country'], inplace=True)
output_path = r"C:\Users\kanmani\Desktop\AutoStack360\Data\Cleaned\df_with_country_encoded.csv"
df.to_csv(output_path, index=False)
print(f"Data saved successfully to: {output_path}")

plt.figure(figsize=(8, 5))
sns.regplot(x='Country_Encoded', y='LogSalary', data=df, scatter_kws={'alpha':0.3})
plt.title("Correlation between Country_Encoded and LogSalary")
plt.xlabel("Country_Encoded")
plt.ylabel("LogSalary")
plt.tight_layout()
plt.show()
