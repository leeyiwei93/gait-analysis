from matplotlib import pyplot as plt
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn import metrics

file1 = os.path.join('SpinData2.csv')#'D:\Python\mediapipe\SpinData.csv' ************
file2 = os.path.join('spinning2.pkl')#'D:\Python\mediapipe\model.pkl file' *********

df = pd.read_csv(file1)# ************

X = df.drop('class', axis=1) # features
y = df['class'] # target value

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

pipelines = {
    'lr':make_pipeline(StandardScaler(), LogisticRegression(max_iter=200)),
    'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
    'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
    'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
}

fit_models = {}
for algo, pipeline in pipelines.items():
    model = pipeline.fit(X_train, y_train)
    fit_models[algo] = model

for algo, model in fit_models.items():
    y_pred = model.predict(X_test)
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")
    plt.legend()
    plt.show()
    print(algo, accuracy_score(y_test, y_pred))

with open(file2, 'wb') as f: # *************
    pickle.dump(fit_models['rf'], f) # ************* choose the highest accuracy

with open(file2, 'rb') as f: # ***********
    model = pickle.load(f)




