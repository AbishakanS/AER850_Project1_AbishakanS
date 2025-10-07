# AER 850: Project 1
# Abishakan Shivananthan
# 501102402
# Section 2

# importing needed libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Data Processing
csv_file = 'Project 1 Data.csv'
df = pd.read_csv(csv_file)
df = df.dropna().reset_index(drop=True) # Remove missing data

# Step 2: Data Visualization

#Basic Data Info and summary
print("First few rows of the DataFrame:")
print(df.head())
print("\nDataFrame info:")
print(df.info())
print("\nSummary statistics:")
print(df.describe())
print("\nNumber of samples in each step:")
print(df['Step'].value_counts())
# grouped_data = df.groupby('Step').mean()
# print("\nGrouped DataFrame with mean values:")
# print(grouped_data)


# Step 3: Correlation Analysis

corr_matrix = df.corr(method='pearson') # Compute the correlation matrix using Pearson correlation
print("\nCorrelation matrix:") # Display the correlation values
print(corr_matrix)
sns.heatmap(np.abs(corr_matrix))


# Step 4: Classification Model Development/Engineering
from sklearn.model_selection import StratifiedShuffleSplit

my_splitter = StratifiedShuffleSplit(n_splits=1, 
                                     test_size=0.2, 
                                     random_state=42)

# Split based on 'Step' column
for train_index, test_index in my_splitter.split(df, df["Step"]):
    strat_train = df.loc[train_index].reset_index(drop=True)
    strat_test = df.loc[test_index].reset_index(drop=True)

# Display sizes
print("Training set size:", len(strat_train))
print("Test set size:", len(strat_test))


# Step 5: Model Performance Analysis

# Step 6: Stacked Model Performance Analysis

# Step 7: Model Evaluation


