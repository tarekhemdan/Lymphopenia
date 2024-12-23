import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Assuming the dataset is loaded into a Pandas DataFrame
# Replace 'data.csv' with your dataset file path
df = pd.read_csv('lymphonia_encoded.csv')

# Extract A/C ratio 2 and Lymphocytes 2
ac_ratio_2 = df['A_C ratio 2']  # Replace with actual column name for A/C ratio 2
lymphocytes_2 = df['Lymphocytes 2']  # Replace with actual column name for Lymphocytes 2

# Drop missing values
valid_data = df[['A_C ratio 2', 'Lymphocytes 2']].dropna()

# Scatterplot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=valid_data['A_C ratio 2'], y=valid_data['Lymphocytes 2'])
plt.title('Scatterplot of A/C Ratio 2 vs. Lymphocytes 2')
plt.xlabel('A/C Ratio 2')
plt.ylabel('Lymphocytes 2')
plt.show()

# Correlation
corr, p_value = pearsonr(valid_data['A_C ratio 2'], valid_data['Lymphocytes 2'])
print(f"Pearson Correlation: {corr:.2f}, P-value: {p_value:.2e}")

corr, p_value = pearsonr(valid_data['A_C ratio 2'], valid_data['Lymphocytes 2'])
print(f"Pearson Correlation: {corr:.2f}, P-value: {p_value:.2e}")

# Extract A/C ratio 2 and Lymphocytes 2
ac_ratio_2 = df['A_C ratio 2']  # Here, replace with the actual column name for A/C ratio 2
lymphocytes_2 = df['Lymphocytes 2']  # Here, replace with the actual column name for Lymphocytes 2

# Drop missing values
valid_data = df[['A_C ratio 2', 'Lymphocytes 2']].dropna()

# Scatterplot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=valid_data['A_C ratio 2'], y=valid_data['Lymphocytes 2'])
plt.title('Scatterplot of A/C Ratio 2 vs. Lymphocytes 2')
plt.xlabel('A/C Ratio 2')
plt.ylabel('Lymphocytes 2')
plt.axhline(y=1500, color='red', linestyle='--')  # Indicating lymphopenia threshold
plt.grid(True)
plt.show()

######################################################################################

# Extract A/C ratio 2 and Lymphocytes 2
ac_ratio = df['A_C ratio']  # Here, replace with the actual column name for A/C ratio 2
lymphocytes = df['lymphocytes']  # Here, replace with the actual column name for Lymphocytes 2

# Drop missing values
valid_data = df[['A_C ratio', 'lymphocytes']].dropna()
print("============ok==============")
# Scatterplot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=valid_data['A_C ratio'], y=valid_data['lymphocytes'])
plt.title('Scatterplot of A/C Ratio vs. Lymphocytes')
plt.xlabel('A/C Ratio')
plt.ylabel('Lymphocytes')
plt.axhline(y=1500, color='red', linestyle='--')  # Indicating lymphopenia threshold
plt.grid(True)
plt.show()

# Correlation
corr, p_value = pearsonr(valid_data['A_C ratio'], valid_data['lymphocytes'])
print(f"Pearson Correlation: {corr:.2f}, P-value: {p_value:.2e}")

corr, p_value = pearsonr(valid_data['A_C ratio'], valid_data['lymphocytes'])
#Pearson Correlation: -0.02, P-value: 8.87e-01
print(f"Pearson Correlation: {corr:.2f}, P-value: {p_value:.2e}")
