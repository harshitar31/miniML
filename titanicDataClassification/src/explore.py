import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/titanic.csv")
print(df.shape)
print(df.head())
print(df.isnull().sum())

cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
cols = [i for i in cols if df[i].dtype!='object']
fig,ax = plt.subplots(len(cols),len(cols),figsize=(6*len(cols), 6 * len(cols)))
k=0
l=0
for i in cols:
    k=0
    for j in cols:
        ax[l,k].scatter(df[i],df[j],c=df['Survived'])
        ax[l,k].set_xlabel(i)
        ax[l,k].set_ylabel(j)
        k+=1
    l+=1
    
plt.tight_layout()
plt.savefig("src/eda.png")