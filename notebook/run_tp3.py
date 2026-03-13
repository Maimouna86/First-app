import pandas as pd
import numpy as np
import os
print('TP3 Cleaning - Génère loan_data_clean.csv')
df = pd.read_csv('../data/loan_data.csv')
print('Raw data:', df.shape)
df = df.drop('Loan_ID', axis=1, errors='ignore')
# Imputation
num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
for col in num_cols:
    if col in df:
        df[col] = df[col].fillna(df[col].median())
for col in cat_cols:
    if col in df:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
# Outlier capping
def cap(df, col, p1=1, p99=99):
    low = df[col].quantile(p1/100)
    high = df[col].quantile(p99/100)
    df[col] = df[col].clip(low, high)
for col in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']:
    if col in df:
        cap(df, col)
# Feature engineering
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['LoanAmountToIncome'] = df['LoanAmount'] / (df['TotalIncome'] + 1)
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(360)
df['EMI'] = df['LoanAmount'] / df['Loan_Amount_Term']
df['EMIToIncome'] = df['EMI'] / (df['TotalIncome'] + 1)
df['Log_LoanAmount'] = np.log(df['LoanAmount'] + 1)
df['Log_TotalIncome'] = np.log(df['TotalIncome'] + 1)
df['Has_Coapplicant'] = (df['CoapplicantIncome'] > 0).astype(int)
df['Credit_History'] = df['Credit_History'].fillna(1.0)
# Encoding
df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})
df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
df['Dependents'] = df['Dependents'].replace('3+', 3).astype(float)
df = pd.get_dummies(df, columns=['Gender', 'Married', 'Self_Employed', 'Property_Area'], drop_first=True)
print('Clean data:', df.shape)
os.makedirs('../data/processed', exist_ok=True)
df.to_csv('../data/processed/loan_data_clean.csv', index=False)
print('✅ loan_data_clean.csv créé ! TP4 prêt.')
