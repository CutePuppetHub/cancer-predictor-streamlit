import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

def model_creation(data):
    X = data.drop(['diagnosis'],axis=1)
    y = data['diagnosis']

    scalar = StandardScaler()

    X = scalar.fit_transform(X)

    #split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    #train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("The Accurafcy score of the model is : ", accuracy_score(y_test, y_pred))
    print("The Classification report of the model is : ", classification_report(y_test, y_pred))

    return model, scalar



def get_clean_data():
    data = pd.read_csv("D:/mpl/Streamlit_Project/data/data.csv")
    data = data.drop(['Unnamed: 32', "id"],axis=1)

    data["diagnosis"] = data["diagnosis"].map({'M': 1, 'B' : 0})

    return data


def main():
    data = get_clean_data()
    model , scalar = model_creation(data)
    with open("model.pkl","wb") as f:
        pickle.dump(model,f)
    with open("scalar.pkl","wb") as f:
        pickle.dump(scalar,f)






if __name__ == "__main__":
    main()