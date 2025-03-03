# Import required libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
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

# Define the 5 regression models
models = [ElasticNetCV(), RandomForestRegressor(), SVR(kernel='rbf'), BaggingRegressor(), KNeighborsRegressor()]

# Train and evaluate each model using KFold cross-validation
for model in models:
    start_time = time.time()
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_list, mae_list, r2_list = [], [], []
    for train_index, test_index in kfold.split(X_train):
        X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_test_fold)
        mse_list.append(mean_squared_error(y_test_fold, y_pred))
        mae_list.append(mean_absolute_error(y_test_fold, y_pred))
        r2_list.append(r2_score(y_test_fold, y_pred))
    end_time = time.time()

    # Print the model performance metrics
    print ("Prediction Results foe A/C ratio 2:")
    print(f"Model: {model.__class__.__name__}")
    print(f"Mean squared error: {sum(mse_list)/len(mse_list):.4f}")
    print(f"Mean absolute error: {sum(mae_list)/len(mae_list):.4f}")
    print(f"R-squared score: {sum(r2_list)/len(r2_list):.4f}")
    print(f"Training time: {end_time-start_time:.4f} seconds\n")
    
##########################################################################################
# Import required libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
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

# Define the 5 regression models
models = [ElasticNetCV(), RandomForestRegressor(), SVR(kernel='rbf'), BaggingRegressor(), KNeighborsRegressor()]

# Train and evaluate each model using KFold cross-validation
for model in models:
    start_time = time.time()
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_list, mae_list, r2_list = [], [], []
    for train_index, test_index in kfold.split(X_train):
        X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_test_fold)
        mse_list.append(mean_squared_error(y_test_fold, y_pred))
        mae_list.append(mean_absolute_error(y_test_fold, y_pred))
        r2_list.append(r2_score(y_test_fold, y_pred))
    end_time = time.time()

    # Print the model performance metrics
    print ("Prediction Results for 24h protien 2:")
    print(f"Model: {model.__class__.__name__}")
    print(f"Mean squared error: {sum(mse_list)/len(mse_list):.4f}")
    print(f"Mean absolute error: {sum(mae_list)/len(mae_list):.4f}")
    print(f"R-squared score: {sum(r2_list)/len(r2_list):.4f}")
    print(f"Training time: {end_time-start_time:.4f} seconds\n")
    
    ##########################################################################################
    # Import required libraries
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.model_selection import KFold
    from sklearn.linear_model import ElasticNetCV
    from sklearn.ensemble import BaggingRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.model_selection import train_test_split
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

    # Define the 5 regression models
    models = [ElasticNetCV(), RandomForestRegressor(), SVR(kernel='rbf'), BaggingRegressor(), KNeighborsRegressor()]

    # Train and evaluate each model using KFold cross-validation
    for model in models:
        start_time = time.time()
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        mse_list, mae_list, r2_list = [], [], []
        for train_index, test_index in kfold.split(X_train):
            X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
            y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
            model.fit(X_train_fold, y_train_fold)
            y_pred = model.predict(X_test_fold)
            mse_list.append(mean_squared_error(y_test_fold, y_pred))
            mae_list.append(mean_absolute_error(y_test_fold, y_pred))
            r2_list.append(r2_score(y_test_fold, y_pred))
        end_time = time.time()

        # Print the model performance metrics
        print ("Prediction Results for albumin 2:")
        print(f"Model: {model.__class__.__name__}")
        print(f"Mean squared error: {sum(mse_list)/len(mse_list):.4f}")
        print(f"Mean absolute error: {sum(mae_list)/len(mae_list):.4f}")
        print(f"R-squared score: {sum(r2_list)/len(r2_list):.4f}")
        print(f"Training time: {end_time-start_time:.4f} seconds\n")