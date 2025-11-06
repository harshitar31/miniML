import torch 
import torch.nn as nn
import torch.optim as optim
from preprocess import loadData

class CarPricePredictionModel(nn.Module):
    def __init__(self, inputDim):
        super(CarPricePredictionModel,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(inputDim,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,1)
        )

    def forward(self,x):
        return self.net(x)
    

def trainModel(model,XTrain,yTrain,epochs=200,lr=0.001):
    criterion=nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(XTrain)
        loss = criterion(outputs,yTrain)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}  Loss: {loss.item()}")

if __name__=="__main__":
    path=r"data\raw\CarPrice_Assignment.csv"
    XTrain,XTest,yTrain,yTest = loadData(path)

    XTrainT = torch.tensor(XTrain,dtype=torch.float32)
    yTrainT = torch.tensor(yTrain, dtype=torch.float32).view(-1,1)

    model = CarPricePredictionModel(inputDim=XTrain.shape[1])
    trainModel(model,XTrainT,yTrainT,epochs=200,lr=0.01)
