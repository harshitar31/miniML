import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"data/raw/CarPrice_Assignment.csv")

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

#To find correlation
cols = [i for i in df.columns if df[i].dtype != "object"]
fig,ax = plt.subplots(len(cols),1,figsize=(6, 6 * len(cols)))
k=0
for i in cols:
    ax[k].scatter(df[i],df["price"])
    ax[k].set_xlabel(i)
    ax[k].set_ylabel("price")
    k+=1
    
plt.tight_layout()
plt.savefig("src/eda.png")
