import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def loadData(path):
    df = pd.read_csv(path)
    print("Original Shape: ", df.shape)
    df = df.drop(columns=['car_ID','CarName'])
    
    X = df.drop(columns=['price'])
    y = df['price']

    X = pd.get_dummies(X,drop_first=True)

    XTrain,XTest,yTrain,yTest = train_test_split(X,y,test_size=0.2,random_state=40)

    scaler = StandardScaler()
    XTrainScaled = scaler.fit_transform(XTrain)
    XTestScaled = scaler.transform(XTest)

    print("Train shape:", XTrainScaled.shape)
    print("Test shape:", XTestScaled.shape)

    return XTrainScaled, XTestScaled, yTrain.values, yTest.values


if __name__ == "__main__":
    path=r"data\raw\CarPrice_Assignment.csv"
    XTrain,XTest,yTrain,yTest = loadData(path)

    
