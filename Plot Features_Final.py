import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
adni_df = pd.read_csv('lymphonia_encoded.csv')

# Identify and clean non-numeric columns
for col in adni_df.columns:
    if adni_df[col].dtype == 'object':  # Check for string columns
        adni_df[col] = adni_df[col].str.replace(',', '')  # Remove commas
        adni_df[col] = pd.to_numeric(adni_df[col], errors='coerce')  # Convert to numeric

# Handle missing values
adni_df.fillna(adni_df.median(), inplace=True)

# Separate features and target variable
target_col = 'albumin 2'  # Replace with your target column name
features = adni_df.drop(columns=[target_col])  # Exclude target variable

# Visualizations
sns.set_style('darkgrid')
sns.set_palette('muted')

# 1. Plot histograms of features
for col in features.columns:
    try:
        fig, ax = plt.subplots()
        sns.histplot(data=adni_df, x=col, hue=target_col, kde=True, ax=ax)
        ax.set_title(f"Histogram of {col}")
        plt.savefig(f"{col}_histogram.png", dpi=300, bbox_inches='tight')
        plt.show()
    except Exception as e:
        print(f"Skipping histogram for {col} due to error: {e}")

# 2. Plot boxplots of features
for col in features.columns:
    try:
        fig, ax = plt.subplots()
        sns.boxplot(data=adni_df, x=target_col, y=col, ax=ax)
        ax.set_title(f"Boxplot of {col}")
        plt.savefig(f"{col}_boxplot.png", dpi=300, bbox_inches='tight')
        plt.show()
    except Exception as e:
        print(f"Skipping boxplot for {col} due to error: {e}")

# 3. Plot the correlation matrix of features
try:
    corr_matrix = features.corr()
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title("Correlation Matrix of Features")
    plt.savefig("correlation_matrix.png", dpi=300, bbox_inches='tight')
    plt.show()
except Exception as e:
    print(f"Skipping correlation matrix due to error: {e}")
    
"""
# 4. Pairplot of features
try:
    sns.pairplot(adni_df, hue=target_col, diag_kind="kde", corner=True)
    plt.savefig("pairplot.png", dpi=300, bbox_inches='tight')
    plt.show()
except Exception as e:
    print(f"Skipping pairplot due to error: {e}")
"""
# 5. Violin plots for features
for col in features.columns:
    try:
        fig, ax = plt.subplots()
        sns.violinplot(data=adni_df, x=target_col, y=col, ax=ax)
        ax.set_title(f"Violin plot of {col}")
        plt.savefig(f"{col}_violinplot.png", dpi=300, bbox_inches='tight')
        plt.show()
    except Exception as e:
        print(f"Skipping violin plot for {col} due to error: {e}")

# 6. Scatterplots of features against target variable
for col in features.columns:
    try:
        fig, ax = plt.subplots()
        sns.scatterplot(data=adni_df, x=col, y=target_col, ax=ax)
        ax.set_title(f"Scatter plot of {col} vs. {target_col}")
        plt.savefig(f"{col}_scatterplot.png", dpi=300, bbox_inches='tight')
        plt.show()
    except Exception as e:
        print(f"Skipping scatter plot for {col} due to error: {e}")

print("Visualization completed successfully.")
