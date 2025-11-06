import torch
import numpy as np
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from preprocess import loadData
from app import CarPricePredictionModel,trainModel
import os 


os.makedirs("results",exist_ok=True)
path=r"data\raw\CarPrice_Assignment.csv"
XTrain,XTest,yTrain,yTest = loadData(path)

XTrainT = torch.tensor(XTrain,dtype=torch.float32)
yTrainT = torch.tensor(yTrain, dtype=torch.float32).view(-1,1)
XTestT = torch.tensor(XTest,dtype=torch.float32)
yTestnp = np.array(yTest).reshape(-1)

model = CarPricePredictionModel(inputDim=XTrain.shape[1])
trainModel(model,XTrainT,yTrainT,epochs=100,lr=0.01)

model.eval()
with torch.no_grad():
    predT = model(XTestT).cpu().numpy().reshape(-1)
pred = predT 

rmse = root_mean_squared_error(yTestnp,pred)
mae = mean_absolute_error(yTestnp,pred)

print(f"RMSE: {rmse:.2f}")
print(f"MAE:  {mae:.2f}")

print("First 5 predictions")
for i in range(5):
    print(f"pred={pred[i]}  true={yTestnp[i]}")

plt.figure(figsize=(8,8))
plt.scatter(yTestnp,pred)
plt.xlabel("True Price")
plt.ylabel("Predicted Price")
plt.title("Predicted vs True Car Price")
plt.axis('equal')
plt.plot([yTestnp.min(),yTestnp.max()],[yTestnp.min(),yTestnp.max()],'g--')
plt.savefig("results/predVsTrue.png")

print("Scatter plot figure saved in results folder")
           
