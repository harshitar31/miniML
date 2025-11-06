import pandas as pd
import numpy as np

df = pd.read_csv(r"data\raw\CarPrice_Assignment.csv")

print("First 5 Rows")
print(df.head())
print()

print("Description")
print(df.describe().T)
print()

print("Null Values")
print(df.isnull().sum())
print()

print("Info")
print(df.info())