import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def show_result(clf, model_name, X_test, y_test):
    y_pred = clf.predict(X_test)
    if model_name != "SVC Model":
        y_probs = clf.predict_proba(X_test)
    print("\n"+model_name)
    print("F1 Score: ", f1_score(y_test, y_pred))
    if model_name != "SVC Model":
        print("AUC: ", roc_auc_score(y_test, y_probs[:,1]))
    print("Confusion Matrix: \n",confusion_matrix(y_test, y_pred))

def logistic_model(X_train, X_test, y_train, y_test):
    clf = LogisticRegression(C=100).fit(X_train, y_train)
    show_result(clf, "Logistic Regression Model", X_test, y_test)
    
def svc_model(X_train, X_test, y_train, y_test):
    clf = SVC(C=100).fit(X_train, y_train)
    show_result(clf, "SVC Model", X_test, y_test)
    
def decision_model(X_train, X_test, y_train, y_test):
    clf = DecisionTreeClassifier().fit(X_train, y_train)
    show_result(clf, "Decision Tree Model", X_test, y_test)
    
def xgboost_model(X_train, X_test, y_train, y_test):
    clf = GradientBoostingClassifier().fit(X_train, y_train)
    show_result(clf, "XGBoost Model", X_test, y_test)
    
def neuralnetwork_model(X_train, X_test, y_train, y_test):
    clf = MLPClassifier(hidden_layer_sizes=(20, 20, 20),max_iter=1000).fit(X_train, y_train)
    show_result(clf, "Neural Network Model", X_test, y_test)
    
def randomforest_model(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier().fit(X_train, y_train)
    show_result(clf, "Random Forest Model", X_test, y_test)
    
if __name__ == "__main__":

    data = pd.read_csv("data.csv") 
    y_data = data['label']
    y_data = LabelEncoder().fit_transform(y_data)
    X_data = data.drop(['label'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
    
    logistic_model(X_train, X_test, y_train, y_test)
    svc_model(X_train, X_test, y_train, y_test)
    decision_model(X_train, X_test, y_train, y_test)
    xgboost_model(X_train, X_test, y_train, y_test)
    neuralnetwork_model(X_train, X_test, y_train, y_test)
    randomforest_model(X_train, X_test, y_train, y_test)
    
