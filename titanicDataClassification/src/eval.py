import torch
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from preprocess import loadData
from app import titanicClassificationModel,trainModel
import os 


os.makedirs("results",exist_ok=True)
path=r"data/titanic.csv"
XTrain,XTest,yTrain,yTest = loadData(path)

XTrainT = torch.tensor(XTrain,dtype=torch.float32)
yTrainT = torch.tensor(yTrain, dtype=torch.float32).view(-1,1)
XTestT = torch.tensor(XTest,dtype=torch.float32)
yTestnp = np.array(yTest).reshape(-1)

model = titanicClassificationModel(inputDim=XTrain.shape[1])
trainModel(model,XTrainT,yTrainT,epochs=100,lr=0.01)

model.eval()
with torch.no_grad():
    logits = model(XTestT)
    probs = torch.sigmoid(logits)
    preds = (probs>=0.5).float()

print("First 5 predictions")
for i in range(5):
    print(f"pred={preds[i]}  true={yTestnp[i]}")

accuracy = accuracy_score(yTest,preds)
print(f'Accuracy score: {accuracy}')           
