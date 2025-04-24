import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

# Load the dataset
file_path = '/content/bf_dataset.xlsx'
data = pd.read_excel(file_path)

# Convert columns to numeric, coercing errors to NaN
for column in data.columns:
    data[column] = pd.to_numeric(data[column], errors='coerce')

# Handle missing values by filling NaN with the column mean
data.fillna(data.mean(), inplace=True)

# Feature engineering: Calculate CO_CO2_Ratio
data['CO_CO2_Ratio'] = data['CO'] / data['CO2']

# Extract features and target variable
X = data.drop(columns=['CO2', 'CO', 'CO_CO2_Ratio'])
y = data[['CO', 'CO2', 'CO_CO2_Ratio']]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Impute missing values if any
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models dictionary for performance comparison
models = {
    'Linear Regression': LinearRegression(),
    'SVR': SVR(),
    'KNN Regression': KNeighborsRegressor(),
    'Random Forest': RandomForestRegressor(),
    'Naive Bayes (converted)': GaussianNB(),  # Classification-style approximation
    'ANN': MLPRegressor(max_iter=1000),
    'XGBoost': xgb.XGBRegressor()
}

# Train and evaluate models
results = {}

for name, model in models.items():
    if name == 'Naive Bayes (converted)':
        # Naive Bayes is a classification model, so we use a binary transformation here
        model.fit(X_train_scaled, y_train['CO'] > y_train['CO'].mean())  # Example: CO prediction as binary classification
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error((y_test['CO'] > y_train['CO'].mean()).astype(int), y_pred)
    else:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
    
    # Record the model results
    results[name] = mse
    print(f'{name} - MSE: {mse}')

# Sort and display the model performance based on MSE
results_sorted = dict(sorted(results.items(), key=lambda x: x[1]))
print("Model Comparison (Lower MSE is Better):")
for name, mse in results_sorted.items():
    print(f"{name}: {mse:.4f}")

# Display scatter plot for the best model's predictions vs. actual values
best_model_name = min(results, key=results.get)
best_model = models[best_model_name]
plt.figure(figsize=(10, 6))
plt.scatter(y_test['CO_CO2_Ratio'], best_model.predict(X_test_scaled)[:, 2], alpha=0.7)
plt.xlabel('Actual CO/CO2 Ratio')
plt.ylabel('Predicted CO/CO2 Ratio')
plt.title(f'Actual vs Predicted CO/CO2 Ratio using {best_model_name}')
plt.plot([min(y_test['CO_CO2_Ratio']), max(y_test['CO_CO2_Ratio'])], [min(y_test['CO_CO2_Ratio']), max(y_test['CO_CO2_Ratio'])], color='red', lw=2)
plt.show()

# Predict using new sample data
new_data = np.array([[315163, 3.16, 129, 4, 209, 3.35, 3.2, 7829, 23.08, 30, 24.52, 1058, 2.99, 1.49, 120, 143, 109, 128, 0, 125, 1, 3.94, 71.5]])
new_data_scaled = scaler.transform(new_data)
new_data_pred = best_model.predict(new_data_scaled)

print(f'Predicted CO: {new_data_pred[0][0]}')
print(f'Predicted CO2: {new_data_pred[0][1]}')
print(f'Predicted CO/CO2 Ratio: {new_data_pred[0][2]}')
