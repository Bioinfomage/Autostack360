
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer

# Setup for better visuals
pd.set_option('display.max_columns', None)

# Load the cleaned dataset
df = pd.read_csv("C:/Users/kanmani/Desktop/AutoStack360/Data/Cleaned/df_with_country_encoded.csv")

# Print unique values in EdLevel
print(df['EdLevel'].unique())

# Simplify education levels
df['EdLevel'] = df['EdLevel'].replace({
    "Bachelor’s degree (B.A., B.S., B.Eng., etc.)": "Bachelor’s",
    "Master’s degree (M.A., M.S., M.Eng., MBA, etc.)": "Master’s",
    "Some college/university study without earning a degree": "Some College",
    "Professional degree (JD, MD, Ph.D, Ed.D, etc.)": "Professional",
    "Associate degree (A.A., A.S., etc.)": "Associate",
    "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)": "Secondary School",
    "Primary/elementary school": "Primary School",
    "Something else": "Other"
})

# Boxplot for LogSalary by EdLevel
plt.figure(figsize=(10, 6))
sns.boxplot(
    x='EdLevel', y='LogSalary', data=df,
    order=[
        "Primary School", "Secondary School", "Some College",
        "Associate", "Bachelor’s", "Master’s", "Professional", "Other"
    ]
)
plt.xticks(rotation=45)
plt.title("LogSalary by Education Level")
plt.xlabel("Education Level")
plt.ylabel("Log of Salary")
plt.tight_layout()
plt.show()

# One-Hot Encode EdLevel
df = pd.get_dummies(df, columns=['EdLevel'], drop_first=True)

# Correlation with LogSalary for EdLevel
edlevel_cols = [col for col in df.columns if col.startswith('EdLevel_')]
correlations = df[edlevel_cols + ['LogSalary']].corr()['LogSalary'].drop('LogSalary')
print(correlations.sort_values(ascending=False))

# Preprocess LanguageHaveWorkedWith
df['LanguageHaveWorkedWith'] = df['LanguageHaveWorkedWith'].fillna('').astype(str)
df['LanguageHaveWorkedWith'] = df['LanguageHaveWorkedWith'].apply(lambda x: x.split(';') if x.strip() != '' else [])

# MultiLabelBinarizer
mlb = MultiLabelBinarizer()
language_dummies = pd.DataFrame(
    mlb.fit_transform(df['LanguageHaveWorkedWith']),
    columns=mlb.classes_,
    index=df.index
)

# Concatenate language dummies
df = pd.concat([df, language_dummies], axis=1)

# Correlation with LogSalary for languages
language_cols = mlb.classes_
language_corr = df[list(language_cols) + ['LogSalary']].corr()['LogSalary'].drop('LogSalary')
print(language_corr.sort_values(ascending=False))

# Top 10 languages
top_langs = language_corr.abs().sort_values(ascending=False).head(10).index
plt.figure(figsize=(10, 5))
sns.barplot(x=language_corr[top_langs].values, y=top_langs)
plt.title("Top 10 Language Correlations with LogSalary")
plt.xlabel("Correlation with LogSalary")
plt.tight_layout()
plt.show()

# Drop original language column and save
df.drop(columns=['LanguageHaveWorkedWith'], inplace=True)
output_path = r"C:\Users\kanmani\Desktop\AutoStack360\Data\Cleaned\survey_with_language_nd_Edlevel_binaries.csv"
df.to_csv(output_path, index=False)

print("File saved successfully at:")
print(output_path)
