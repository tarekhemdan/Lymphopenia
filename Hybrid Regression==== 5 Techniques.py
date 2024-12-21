"""This program loads the ADNI dataset as a Pandas dataframe, splits the data into features 
and target, scales the features using StandardScaler, splits the data into training and testing
 sets, trains Random Forest, SVM, K-Nearest Neighbors, Decision Tree, and Gradient Boosting 
 models, combines the predictions from all models using a simple averaging method, evaluates 
 the hybrid model with the testing set, and prints out the Mean Squared Error (MSE), 
 Mean Absolute Error (MAE), and R-squared (R^2) score for the hybrid model.
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time

start_time = time.time()
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



# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train with Random Forest
rf_model = RandomForestRegressor(n_estimators=100)
rf_model.fit(X_train, y_train)

# Train with SVM
svm_model = SVR()
svm_model.fit(X_train, y_train)

# Train with K-Nearest Neighbors
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Train with Decision Tree
dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, y_train)

# Train with Gradient Boosting
gb_model = GradientBoostingRegressor(n_estimators=100)
gb_model.fit(X_train, y_train)

# Combine predictions from all models
rf_pred = rf_model.predict(X_test)
svm_pred = svm_model.predict(X_test)
knn_pred = knn_model.predict(X_test)
dt_pred = dt_model.predict(X_test)
gb_pred = gb_model.predict(X_test)

# Combine predictions from all models
hybrid_pred = (rf_pred + svm_pred + knn_pred + dt_pred + gb_pred) / 5
end_time = time.time()

# Evaluate hybrid model with testing set
print("Hybrid Regression Metrics A/C ratio 2:")
print("Mean Squared Error (MSE):", mean_squared_error(y_test, hybrid_pred))
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, hybrid_pred))
print("R-squared (R^2) Score:", r2_score(y_test, hybrid_pred))

# Print execution time
execution_time = end_time - start_time
print("Execution Time:", execution_time)
print()

#####################################################################################

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time

start_time = time.time()
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


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train with Random Forest
rf_model = RandomForestRegressor(n_estimators=100)
rf_model.fit(X_train, y_train)

# Train with SVM
svm_model = SVR()
svm_model.fit(X_train, y_train)

# Train with K-Nearest Neighbors
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Train with Decision Tree
dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, y_train)

# Train with Gradient Boosting
gb_model = GradientBoostingRegressor(n_estimators=100)
gb_model.fit(X_train, y_train)

# Combine predictions from all models
rf_pred = rf_model.predict(X_test)
svm_pred = svm_model.predict(X_test)
knn_pred = knn_model.predict(X_test)
dt_pred = dt_model.predict(X_test)
gb_pred = gb_model.predict(X_test)

# Combine predictions from all models
hybrid_pred = (rf_pred + svm_pred + knn_pred + dt_pred + gb_pred) / 5
end_time = time.time()

# Evaluate hybrid model with testing set
print("Hybrid Regression Metrics for 24h protien 2:")
print("Mean Squared Error (MSE):", mean_squared_error(y_test, hybrid_pred))
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, hybrid_pred))
print("R-squared (R^2) Score:", r2_score(y_test, hybrid_pred))

# Print execution time
execution_time = end_time - start_time
print("Execution Time:", execution_time)
print()

#####################################################################################

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time

start_time = time.time()
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


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train with Random Forest
rf_model = RandomForestRegressor(n_estimators=100)
rf_model.fit(X_train, y_train)

# Train with SVM
svm_model = SVR()
svm_model.fit(X_train, y_train)

# Train with K-Nearest Neighbors
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Train with Decision Tree
dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, y_train)

# Train with Gradient Boosting
gb_model = GradientBoostingRegressor(n_estimators=100)
gb_model.fit(X_train, y_train)

# Combine predictions from all models
rf_pred = rf_model.predict(X_test)
svm_pred = svm_model.predict(X_test)
knn_pred = knn_model.predict(X_test)
dt_pred = dt_model.predict(X_test)
gb_pred = gb_model.predict(X_test)

# Combine predictions from all models
hybrid_pred = (rf_pred + svm_pred + knn_pred + dt_pred + gb_pred) / 5
end_time = time.time()

# Evaluate hybrid model with testing set
print("Hybrid Regression Metrics for albumin 2:")
print("Mean Squared Error (MSE):", mean_squared_error(y_test, hybrid_pred))
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, hybrid_pred))
print("R-squared (R^2) Score:", r2_score(y_test, hybrid_pred))

# Print execution time
execution_time = end_time - start_time
print("Execution Time:", execution_time)
print()
