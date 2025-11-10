import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def loadData(path):
    df = pd.read_csv(path)
    df=df.drop(columns=['PassengerId','Name','Ticket','Cabin'])
    print(df.head())
    df['Age']=df['Age'].fillna(df['Age'].median())
    df['Embarked']= df['Embarked'].fillna(df['Embarked'].mode()[0])

    df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
    df = pd.get_dummies(df,columns=['Embarked'],drop_first=True)

    X = df.drop(columns=['Survived']).values
    y = df['Survived'].values 

    XTrain, XTest, yTrain, yTest = train_test_split(X,y,test_size=0.2,random_state=50)

    scaler = StandardScaler()
    XTrainScaled = scaler.fit_transform(XTrain)
    XTestScaled = scaler.transform(XTest)

    print(df.columns)

    return XTrainScaled, XTestScaled, yTrain, yTest

if __name__ == '__main__':
    path = 'data/titanic.csv'
    XTrain, XTest, yTrain, yTest = loadData(path)       
    print(XTrain.shape,XTest.shape,yTrain.shape,yTest.shape)
    