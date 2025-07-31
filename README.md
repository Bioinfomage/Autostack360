# Autosack360
A complete data science pipeline to predict developer salaries using survey data.

### 🔧 Data Wrangling & EDA Summary

- Loaded and inspected the Stack Overflow 2024 Developer Survey dataset.
- Cleaned missing values and dropped irrelevant/sparse columns.
- Log-transformed and trimmed extreme salary values for better normalization.
- Simplified education levels (`EdLevel`) and converted to one-hot encoded variables.
- Transformed multi-valued column `LanguageHaveWorkedWith` using `MultiLabelBinarizer`.
- Applied leakage-free KFold target encoding for `Country` → `Country_Encoded`.
- Removed original categorical columns after encoding to avoid duplication.
- Conducted exploratory analysis of:
  - Salary distribution (raw vs. log)
  - Salary vs. Experience (`YearsCodePro`)
  - Salary trends by `Country`, `EdLevel`, and languages
  - Correlation heatmaps and bar plots for key categorical indicators
## 📁 Project Structure
- `Data/` – cleaned and transformed data
- `Scripts/` – preprocessing, feature engineering, modeling scripts
- `Notebooks/` – EDA and modeling in Jupyter
- `Models/` – saved ML models
- `Outputs/` – predictions and metrics
- `PowerBI_Dashboard/` – reports and visuals
### 🤖 Model Summary

- Model Type: Random Forest Regressor
- Objective: Predict developers' `LogSalary` based on experience, education level, programming languages, and country.
- Features Used:
- `YearsCodePro` (numerical)
  - `Country_Encoded` (leakage-free target encoding)
  - One-hot encoded `EdLevel_*` variables
  - Binarized programming languages (e.g., Python, JavaScript, SQL)
- Train/Test Split:80/20
- Performance Metrics:
  - RMSE: ~0.58
  - R² Score: ~0.57
- Artifacts Saved:
  - reduced_random_forest_salary_model.pkl` – Trained model file
  - salary_predictions_random_forest.csv` – Actual vs. predicted log salaries

✅ The model performed best among tested approaches (Linear, Ridge, Random Forest) and is ready for further deployment or integration into dashboards.
  

## 💡 Tools Used
- Python (Pandas, Scikit-learn, Seaborn)
- Power BI
- Git & GitHub

---
Note: The raw data is not attached here considering it's size.
_This project is part of my portfolio to demonstrate end-to-end data science capabilities._
