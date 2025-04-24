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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

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
X = data.drop(columns=['CO2', 'CO','Iron_ore','LimeStone','Coke','CO_CO2_Ratio'])
y = data[['CO', 'CO2','Iron_ore','LimeStone','Coke','CO_CO2_Ratio']]

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

# --- Adding Transformer Model (using PyTorch) ---
class TransformerModel(nn.Module):
    def __init__(self, input_dim, n_heads=8, hidden_dim=64, num_layers=2, output_dim=3):
        super(TransformerModel, self).__init__()
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=n_heads),
            num_layers=num_layers
        )
        
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = x.permute(1, 0, 2)  # Transformer expects input of shape (seq_len, batch_size, input_dim)
        x = self.encoder(x)
        x = x.mean(dim=0)  # Take the mean of the outputs along the sequence dimension
        return self.fc(x)

# --- Adding LSTM Model (using PyTorch) ---
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=3):
        super(LSTMModel, self).__init__()
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

# Preparing data for Transformer and LSTM models
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

train_data = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# Instantiate and train the Transformer and LSTM models
transformer_model = TransformerModel(input_dim=X_train_scaled.shape[1])
lstm_model = LSTMModel(input_dim=X_train_scaled.shape[1])

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer_transformer = torch.optim.Adam(transformer_model.parameters(), lr=1e-4)
optimizer_lstm = torch.optim.Adam(lstm_model.parameters(), lr=1e-4)

# Training the models
n_epochs = 100
for epoch in range(n_epochs):
    transformer_model.train()
    lstm_model.train()
    
    for X_batch, y_batch in train_loader:
        # Train Transformer model
        optimizer_transformer.zero_grad()
        output_transformer = transformer_model(X_batch)
        loss_transformer = criterion(output_transformer, y_batch)
        loss_transformer.backward()
        optimizer_transformer.step()
        
        # Train LSTM model
        optimizer_lstm.zero_grad()
        output_lstm = lstm_model(X_batch)
        loss_lstm = criterion(output_lstm, y_batch)
        loss_lstm.backward()
        optimizer_lstm.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{n_epochs}], Transformer Loss: {loss_transformer.item():.4f}, LSTM Loss: {loss_lstm.item():.4f}')

# Evaluating the Transformer and LSTM models
transformer_model.eval()
lstm_model.eval()

with torch.no_grad():
    y_pred_transformer = transformer_model(X_test_tensor)
    y_pred_lstm = lstm_model(X_test_tensor)
    
    mse_transformer = mean_squared_error(y_test_tensor, y_pred_transformer)
    mse_lstm = mean_squared_error(y_test_tensor, y_pred_lstm)
    
    print(f'Transformer Model - MSE: {mse_transformer:.4f}')
    print(f'LSTM Model - MSE: {mse_lstm:.4f}')
    
    results['Transformer'] = mse_transformer
    results['LSTM'] = mse_lstm

# Sort and display the model performance including Transformer and LSTM
results_sorted = dict(sorted(results.items(), key=lambda x: x[1]))
print("Model Comparison (Lower MSE is Better):")
for name, mse in results_sorted.items():
    print(f"{name}: {mse:.4f}")

# Display scatter plot for the best model's predictions vs. actual values
best_model_name = min(results, key=results.get)
best_model = models.get(best_model_name, transformer_model)
plt.figure(figsize=(10, 6))
if best_model_name == 'Transformer':
    y_pred_best_model = y_pred_transformer
elif best_model_name == 'LSTM':
    y_pred_best_model = y_pred_lstm
else:
    y_pred_best_model = best_model.predict(X_test_scaled)
plt.scatter(y_test['CO_CO2_Ratio'], y_pred_best_model[:, 2], alpha=0.7)
plt.xlabel('Actual CO/CO2 Ratio')
plt.ylabel('Predicted CO/CO2 Ratio')
plt.title(f'Actual vs Predicted CO/CO2 Ratio using {best_model_name}')
plt.plot([min(y_test['CO_CO2_Ratio']), max(y_test
