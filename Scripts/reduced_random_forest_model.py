
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load preprocessed data
df = pd.read_csv("C:/Users/kanmani/Desktop/AutoStack360/Data/Cleaned/survey_with_language_nd_Edlevel_binaries.csv")

# Select features (update this list if needed)
features = [col for col in df.columns if (
    col.startswith('EdLevel_') or
    col.startswith('Country_Encoded') or
    col.startswith('YearsCodePro') or
    col in ['YearsCodePro', 'Country_Encoded']
    or col in ['Python', 'JavaScript', 'Java', 'C#', 'C++', 'Go', 'Rust', 'SQL']  # include most common language binaries
)]

X = df[features].dropna()
y = df.loc[X.index, 'LogSalary']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.3f}")
print(f"R² Score: {r2:.3f}")

# Save model
joblib.dump(model, "reduced_random_forest_salary_model.pkl")

# Save predictions
df_preds = df.loc[X_test.index, ['LogSalary']].copy()
df_preds['Predicted_LogSalary'] = y_pred
df_preds.to_csv("salary_predictions_random_forest.csv", index=False)

print("Model and predictions saved successfully.")
