import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
print("✅ Sklearn imports réussis")
print("Pandas version:", pd.__version__)
print("Numpy version:", np.__version__)
print("Sklearn LogisticRegression:", LogisticRegression().__class__.__name__)
try:
    df = pd.read_csv('../data/loan_data.csv')
    print("✅ Raw data chargé:", df.shape)
    print(df.head())
except Exception as e:
    print("Erreur data:", str(e))
try:
    df_clean = pd.read_csv('loan_data_clean.csv')
    print("✅ Clean data chargé:", df_clean.shape)
except Exception as e:
    print("❌ loan_data_clean.csv manquant:", str(e))
print("TP4 ready once clean data created.")
