import pandas as pd

df = pd.read_csv("data/titanic.csv")
print(df.shape)
print(df.head())
print(df.isnull().sum())