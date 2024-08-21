import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Load the dataset
dataset = pd.read_csv('dataset.csv')

# Splitting the dataset into features and target
X = dataset.drop(columns=['days_lasted'])
y = dataset['days_lasted']

# Splitting into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Training the model
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Square Error: {rmse}")

# Example prediction for new data
new_data = pd.DataFrame({
    'quantity_received': [1500],
    'daily_consumption_rate': [60],
    'consumption_variability': [1.1],
    'reorder_frequency': [20],
    'safety_stock': [300],
    'lead_time': [10],
    'supply_chain_disruption': [0],
    'process_efficiency': [0.95],
    'waste_rate': [0.03],
    'market_demand': [1.1],
    'seasonality': [2],
    'shelf_life': [200],
    'storage_temperature': [20],
    'initial_quality': [0.9],
    'supplier_variability': [0.1],
    'maintenance_frequency': [30],
    'production_batch_size': [400],
    'cost_per_unit': [25],
    'bulk_discount': [0.1]
})

predicted_days = model.predict(new_data)
print(f"Predicted Days: {predicted_days[0]}")
joblib.dump(model, 'model.pkl')