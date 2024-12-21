# Regression of PosttreatmentVisceral fat
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import time
from hyperopt import fmin, tpe, hp, Trials

# Load the dataset
df = pd.read_csv('AD_train.csv')

# Extract the 'sex' column
sex = df['sex']

# Create an instance of LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the 'sex' column
sex_encoded = label_encoder.fit_transform(sex)

# Update the 'sex' column in the DataFrame
df['sex'] = sex_encoded

# Preprocess the dataset
X = df.drop('PosttreatmentVisceral fat', axis=1)
y = df['PosttreatmentVisceral fat']


# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert to pandas DataFrames
X_train = pd.DataFrame(X, columns=df.columns[:-1])
y_train = pd.DataFrame(y, columns=['PosttreatmentVisceral fat'])


# Define the objective function for Hyperopt
def objective(params):
    n_estimators = int(params['n_estimators'])
    max_depth = int(params['max_depth'])
    min_samples_split = int(params['min_samples_split'])
    min_samples_leaf = int(params['min_samples_leaf'])

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

# Define the search space for Hyperopt
space = {
    'n_estimators': hp.quniform('n_estimators', 100, 1000, 100),
    'max_depth': hp.quniform('max_depth', 5, 30, 1),
    'min_samples_split': hp.quniform('min_samples_split', 2, 20, 1),
    'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10, 1)
}

# Optimize using Hyperopt with 100 iterations
trials = Trials()
start_time = time.time()
best = fmin(objective, space, algo=tpe.suggest, max_evals=100, trials=trials)
end_time = time.time()

# Get the best parameters and retrain the model
best_params = {
    'n_estimators': int(best['n_estimators']),
    'max_depth': int(best['max_depth']),
    'min_samples_split': int(best['min_samples_split']),
    'min_samples_leaf': int(best['min_samples_leaf'])
}
best_reg = RandomForestRegressor(**best_params, random_state=42)
best_reg.fit(X_train.values, y_train.values.ravel())

# Predict on the training set using the best regressor
y_pred = best_reg.predict(X_train.values)

# Calculate regression metrics
mse = mean_squared_error(y_train, y_pred)
mae = mean_absolute_error(y_train, y_pred)
r2 = r2_score(y_train, y_pred)

# Print regression metrics and best parameters
print("Regression for PosttreatmentVisceral fat: ")
print("Best Parameters:", best_params)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R2-score:", r2)

# Print execution time
execution_time = end_time - start_time
print("Execution Time:", execution_time)
print()

#####################################################################################
# Regression of Posttreatment Subcutaneous fat
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import time
from hyperopt import fmin, tpe, hp, Trials

# Load the dataset
df = pd.read_csv('AD_train.csv')

# Extract the 'sex' column
sex = df['sex']

# Create an instance of LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the 'sex' column
sex_encoded = label_encoder.fit_transform(sex)

# Update the 'sex' column in the DataFrame
df['sex'] = sex_encoded

# Preprocess the dataset
X = df.drop('Posttreatment Subcutaneous fat', axis=1)
y = df['Posttreatment Subcutaneous fat']


# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert to pandas DataFrames
X_train = pd.DataFrame(X, columns=df.columns[:-1])
y_train = pd.DataFrame(y, columns=['Posttreatment Subcutaneous fat'])


# Define the objective function for Hyperopt
def objective(params):
    n_estimators = int(params['n_estimators'])
    max_depth = int(params['max_depth'])
    min_samples_split = int(params['min_samples_split'])
    min_samples_leaf = int(params['min_samples_leaf'])

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

# Define the search space for Hyperopt
space = {
    'n_estimators': hp.quniform('n_estimators', 100, 1000, 100),
    'max_depth': hp.quniform('max_depth', 5, 30, 1),
    'min_samples_split': hp.quniform('min_samples_split', 2, 20, 1),
    'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10, 1)
}

# Optimize using Hyperopt with 100 iterations
trials = Trials()
start_time = time.time()
best = fmin(objective, space, algo=tpe.suggest, max_evals=100, trials=trials)
end_time = time.time()

# Get the best parameters and retrain the model
best_params = {
    'n_estimators': int(best['n_estimators']),
    'max_depth': int(best['max_depth']),
    'min_samples_split': int(best['min_samples_split']),
    'min_samples_leaf': int(best['min_samples_leaf'])
}
best_reg = RandomForestRegressor(**best_params, random_state=42)
best_reg.fit(X_train.values, y_train.values.ravel())

# Predict on the training set using the best regressor
y_pred = best_reg.predict(X_train.values)

# Calculate regression metrics
mse = mean_squared_error(y_train, y_pred)
mae = mean_absolute_error(y_train, y_pred)
r2 = r2_score(y_train, y_pred)

# Print regression metrics and best parameters
print("Regression for Posttreatment Subcutaneous fat: ")
print("Best Parameters:", best_params)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R2-score:", r2)

# Print execution time
execution_time = end_time - start_time
print("Execution Time:", execution_time)
print()
