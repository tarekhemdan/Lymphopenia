# Regression of A/C ratio 2
import pandas as pd
import optuna
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import time

# Load the dataset
df = pd.read_csv('lymphonia_encoded.csv')

# Preprocess the dataset
X = df.drop('A/C ratio 2', axis=1)
y = df['A/C ratio 2']

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert to pandas DataFrames
X_train = pd.DataFrame(X, columns=df.columns[:-1])
y_train = pd.DataFrame(y, columns=['A/C ratio 2'])

# Define the objective function for Optuna
def objective(trial):
    # Define the parameters to optimize
    n_estimators = trial.suggest_int('n_estimators', 100, 1000, step=100)
    max_depth = trial.suggest_int('max_depth', 5, 30)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)

    # Create the regressor with the suggested parameters
    reg = RandomForestRegressor(n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                random_state=42)

    # Perform cross-validation with 5 folds
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    for train_index, val_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
        reg.fit(X_train_fold, y_train_fold.values.ravel())
        y_val_pred = reg.predict(X_val_fold)
        cv_scores.append(mean_squared_error(y_val_fold, y_val_pred))

    # Calculate the average mean squared error
    mse_avg = sum(cv_scores) / len(cv_scores)

    return mse_avg  # Optimize for mean squared error

# Optimize using Optuna
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
start_time = time.time()
study.optimize(objective, n_trials=100)
end_time = time.time()

# Get the best parameters and retrain the model
best_params = study.best_params
best_reg = RandomForestRegressor(n_estimators=best_params['n_estimators'],
                                 max_depth=best_params['max_depth'],
                                 min_samples_split=best_params['min_samples_split'],
                                 min_samples_leaf=best_params['min_samples_leaf'],
                                 random_state=42)
best_reg.fit(X_train.values, y_train.values.ravel())

# Predict on the test set using the best regressor
y_pred = best_reg.predict(X_train.values)

# Calculate regression metrics
mse = mean_squared_error(y_train, y_pred)
mae = mean_absolute_error(y_train, y_pred)
r2 = r2_score(y_train, y_pred)

# Print regression metrics and best parameters
print("Regression for A/C ratio 2: ")
print("Best Parameters :", best_params)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R2-score:", r2)

# Print execution time
execution_time = end_time - start_time
print("Execution Time:", execution_time)

#####################################################################################
# Regression of 24h Protein 2
import pandas as pd
import optuna
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import time

# Load the dataset
df = pd.read_csv('lymphonia_encoded.csv')

# Preprocess the dataset
X = df.drop('24h protien 2', axis=1)
y = df['24h protien 2']

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert to pandas DataFrames
X_train = pd.DataFrame(X, columns=df.columns[:-1])
y_train = pd.DataFrame(y, columns=['24h protien 2'])

# Define the objective function for Optuna
def objective(trial):
    # Define the parameters to optimize
    n_estimators = trial.suggest_int('n_estimators', 100, 1000, step=100)
    max_depth = trial.suggest_int('max_depth', 5, 30)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)

    # Create the regressor with the suggested parameters
    reg = RandomForestRegressor(n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                random_state=42)

    # Perform cross-validation with 5 folds
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    for train_index, val_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
        reg.fit(X_train_fold, y_train_fold.values.ravel())
        y_val_pred = reg.predict(X_val_fold)
        cv_scores.append(mean_squared_error(y_val_fold, y_val_pred))

    # Calculate the average mean squared error
    mse_avg = sum(cv_scores) / len(cv_scores)

    return mse_avg  # Optimize for mean squared error

# Optimize using Optuna
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
start_time = time.time()
study.optimize(objective, n_trials=100)
end_time = time.time()

# Get the best parameters and retrain the model
best_params = study.best_params
best_reg = RandomForestRegressor(n_estimators=best_params['n_estimators'],
                                 max_depth=best_params['max_depth'],
                                 min_samples_split=best_params['min_samples_split'],
                                 min_samples_leaf=best_params['min_samples_leaf'],
                                 random_state=42)
best_reg.fit(X_train.values, y_train.values.ravel())

# Predict on the test set using the best regressor
y_pred = best_reg.predict(X_train.values)

# Calculate regression metrics
mse = mean_squared_error(y_train, y_pred)
mae = mean_absolute_error(y_train, y_pred)
r2 = r2_score(y_train, y_pred)

# Print regression metrics and best parameters
print("Regression for 24h Protein 2: ")
print("Best Parameters :", best_params)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R2-score:", r2)

# Print execution time
execution_time = end_time - start_time
print("Execution Time:", execution_time)

#####################################################################################
# Regression of Albumin 2
import pandas as pd
import optuna
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import time

# Load the dataset
df = pd.read_csv('lymphonia_encoded.csv')

# Preprocess the dataset
X = df.drop('albumin 2', axis=1)
y = df['albumin 2']

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert to pandas DataFrames
X_train = pd.DataFrame(X, columns=df.columns[:-1])
y_train = pd.DataFrame(y, columns=['albumin 2'])

# Define the objective function for Optuna
def objective(trial):
    # Define the parameters to optimize
    n_estimators = trial.suggest_int('n_estimators', 100, 1000, step=100)
    max_depth = trial.suggest_int('max_depth', 5, 30)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)

    # Create the regressor with the suggested parameters
    reg = RandomForestRegressor(n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                random_state=42)

    # Perform cross-validation with 5 folds
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    for train_index, val_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
        reg.fit(X_train_fold, y_train_fold.values.ravel())
        y_val_pred = reg.predict(X_val_fold)
        cv_scores.append(mean_squared_error(y_val_fold, y_val_pred))

    # Calculate the average mean squared error
    mse_avg = sum(cv_scores) / len(cv_scores)

    return mse_avg  # Optimize for mean squared error

# Optimize using Optuna
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
start_time = time.time()
study.optimize(objective, n_trials=100)
end_time = time.time()

# Get the best parameters and retrain the model
best_params = study.best_params
best_reg = RandomForestRegressor(n_estimators=best_params['n_estimators'],
                                 max_depth=best_params['max_depth'],
                                 min_samples_split=best_params['min_samples_split'],
                                 min_samples_leaf=best_params['min_samples_leaf'],
                                 random_state=42)
best_reg.fit(X_train.values, y_train.values.ravel())

# Predict on the test set using the best regressor
y_pred = best_reg.predict(X_train.values)

# Calculate regression metrics
mse = mean_squared_error(y_train, y_pred)
mae = mean_absolute_error(y_train, y_pred)
r2 = r2_score(y_train, y_pred)

# Print regression metrics and best parameters
print("Regression for Albumin 2: ")
print("Best Parameters :", best_params)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R2-score:", r2)

# Print execution time
execution_time = end_time - start_time
print("Execution Time:", execution_time)
