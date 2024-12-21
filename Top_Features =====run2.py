import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif, chi2, RFE, SelectFromModel, VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load data
dataset = pd.read_csv("lymphonia_encoded.csv")
X = dataset.drop(['sex'], axis=1)
y = dataset['sex']

# Clean dataset by removing commas and converting to numeric
# Assuming all columns in X (except the target 'sex') are numeric
X = X.apply(lambda x: x.replace(',', '') if x.dtype == 'object' else x)
X = X.apply(pd.to_numeric, errors='coerce')  # Convert to numeric, 'coerce' will turn invalid values into NaN

# Check for NaN values and fill them with the mean of the column (or use another strategy)
X.fillna(X.mean(), inplace=True)  # Filling NaN values with column mean

# 1. Univariate feature selection with ANOVA F-value
fvalue_selector = SelectKBest(f_classif, k=5)
X_fvalue = fvalue_selector.fit_transform(X, y)

# 2. Univariate feature selection with mutual information
mi_selector = SelectKBest(mutual_info_classif, k=5)
X_mi = mi_selector.fit_transform(X, y)

# 3. Univariate feature selection with chi-square (commented out)
# chi2_selector = SelectKBest(chi2, k=5)
# X_chi2 = chi2_selector.fit_transform(X, y)

# 4. Recursive feature elimination (RFE) with logistic regression
lr = LogisticRegression(max_iter=10000)  # Increase max_iter for convergence
rfe_selector = RFE(lr, n_features_to_select=5)
X_rfe = rfe_selector.fit_transform(X, y)

# 5. Principal component analysis (PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_scaled)

# 6. Feature selection with random forests
rf = RandomForestClassifier()
sfm_selector = SelectFromModel(rf, max_features=5)
X_sfm = sfm_selector.fit_transform(X, y)

# 7. Variance thresholding
vt_selector = VarianceThreshold(threshold=0.1)
X_vt = vt_selector.fit_transform(X)

# 8. Recursive feature elimination with random forests
rfe_rf_selector = RFE(rf, n_features_to_select=5)
X_rfe_rf = rfe_rf_selector.fit_transform(X, y)

# 9. Feature importance with random forests
rf.fit(X, y)
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
X_fi = X[importances[:5].index]

# Print selected features for each method
print("F-value selector:", X.columns[fvalue_selector.get_support()])
print("=================================================================")

print("Mutual information selector:", X.columns[mi_selector.get_support()])
print("=================================================================")

#print("Chi-squared selector:", X.columns[chi2_selector.get_support()])
#print("=================================================================")

print("RFE with logistic regression:", X.columns[rfe_selector.get_support()])
print("=================================================================")

print("PCA:", pca.explained_variance_ratio_)
print("=================================================================")

print("Select from model with random forests:", X.columns[sfm_selector.get_support()])
print("=================================================================")

print("Variance thresholding:", X.columns[vt_selector.get_support()])
print("=================================================================")

print("RFE with random forests:", X.columns[rfe_rf_selector.get_support()])
print("=================================================================")

print("Feature importance with random forests:", importances[:5].index)
print("=================================================================")
