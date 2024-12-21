import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, normalize
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Filter out warnings
warnings.filterwarnings("ignore")

# Step 1: Load dataset
file_path = 'lymphonia_Fixed.csv'
data = pd.read_csv(file_path)


# Convert 'sex' to continuous values (Binary encoding for Male and Female)
data['sex'] = data['sex'].map({'m': 1, 'f': 0})
data['Anti-DNA'] = data['Anti-DNA'].map({'positive': 1, 'negative': 0})



# Encode categorical features
categorical_cols = ['low complement', 'casts 2', 'casts', 'ANA']
label_encoders = {}

for col in categorical_cols:
    label_encoder = LabelEncoder()
    data[col] = label_encoder.fit_transform(data[col])
    label_encoders[col] = label_encoder  # Store the encoder for future use (optional)

# Step 2: Handle missing values
# Separate numeric and categorical columns
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()

# Impute missing values for numeric columns using the 'mean' strategy
numeric_imputer = SimpleImputer(strategy='mean')
data[numeric_cols] = numeric_imputer.fit_transform(data[numeric_cols])

# Impute missing values for categorical columns using the 'most_frequent' strategy
categorical_imputer = SimpleImputer(strategy='most_frequent')
data[categorical_cols] = categorical_imputer.fit_transform(data[categorical_cols])


# Save the encoded dataset to CSV and Excel
encoded_csv_path = 'lymphonia_encoded.csv'
data.to_csv(encoded_csv_path, index=False)
encoded_excel_path = 'lymphonia_encoded_excel.xlsx'
data.to_excel(encoded_excel_path, index=False, engine='openpyxl')
"""

# Ensure the target column is numeric
target_col = 'A/C ratio 2'
data[target_col] = pd.to_numeric(data[target_col], errors='coerce')  # Convert to numeric, handle errors
data = data.dropna(subset=[target_col])  # Drop rows with NaN values in target


# Step 3: Prepare data for regression
features = data.drop(columns=[target_col])
target = data[target_col]

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Normalize the features
X_train = normalize(X_train)
X_test = normalize(X_test)


# Step 4: Define models and hyperparameter tuning
results = {}

# Helper function to evaluate models
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    start_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    end_time = time.time()
    
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred),
        'Time (s)': end_time - start_time
    }
    results[name] = metrics

# 1. Gradient Boosting with hyperparameter tuning using RandomizedSearchCV
gbr_param_dist = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5],
    'subsample': [0.7, 0.8, 1.0]
}
gbr = GradientBoostingRegressor()
gbr_random = RandomizedSearchCV(gbr, gbr_param_dist, cv=5, n_iter=5, random_state=42, n_jobs=-1)
evaluate_model('GradientBoosting', gbr_random, X_train, X_test, y_train, y_test)

# 2. XGBoost with hyperparameter tuning using RandomizedSearchCV
xgb_param_dist = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5],
    'subsample': [0.7, 0.8, 1.0]
}
xgboost = xgb.XGBRegressor()
xgb_random = RandomizedSearchCV(xgboost, xgb_param_dist, cv=5, n_iter=5, random_state=42, n_jobs=-1)
evaluate_model('XGBoost', xgb_random, X_train, X_test, y_train, y_test)

# 3. LightGBM with hyperparameter tuning using RandomizedSearchCV
lgb_param_dist = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5],
    'subsample': [0.7, 0.8, 1.0]
}
lgbm = lgb.LGBMRegressor()
lgb_random = RandomizedSearchCV(lgbm, lgb_param_dist, cv=5, n_iter=5, random_state=42, n_jobs=-1)
evaluate_model('LightGBM', lgb_random, X_train, X_test, y_train, y_test)

# 4. CatBoost with hyperparameter tuning using RandomizedSearchCV
cb_param_dist = {
    'iterations': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'depth': [3, 4, 5]
}
catboost = cb.CatBoostRegressor(verbose=0)
cb_random = RandomizedSearchCV(catboost, cb_param_dist, cv=5, n_iter=5, random_state=42, n_jobs=-1)
evaluate_model('CatBoost', cb_random, X_train, X_test, y_train, y_test)

# 5. Neural Network (deep learning) model with dropout and additional layers
def create_nn_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dropout(0.3),  # Increased dropout for regularization
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

nn_model = create_nn_model((X_train.shape[1],))
start_time = time.time()
nn_model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)  # Train for 20 epochs
y_pred_nn = nn_model.predict(X_test)
end_time = time.time()

# Metrics for NN model
metrics_nn = {
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_nn)),
    'MAE': mean_absolute_error(y_test, y_pred_nn),  # Corrected line
    'R2': r2_score(y_test, y_pred_nn),
    'Time (s)': end_time - start_time
}
results['NeuralNetwork'] = metrics_nn


# Step 6: Display Results with formatting
print("\n===== Regression Model Evaluation Results =====")
for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")

# Step 7: Visualize Model Performance
model_names = list(results.keys())
rmse_values = [metrics['RMSE'] for metrics in results.values()]

plt.figure(figsize=(10, 6))
plt.barh(model_names, rmse_values, color='teal')
plt.xlabel('RMSE')
plt.title('Model Comparison (RMSE)')
plt.grid(True)
plt.show()
"""