import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import GridSearchCV
import joblib

df = pd.read_csv('https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv')

X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [5,50,100,150,200],
    'max_depth': [None, 3, 6, 7]
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

mlflow.set_experiment('diabetes-rf-hp')

with mlflow.start_run(description="Best hyperparameter trained RF model") as parent:
    grid_search.fit(X_train, y_train)

    # log all the children 
    for i in range(len(grid_search.cv_results_['params'])):
        print(i)
        with mlflow.start_run(nested=True) as child:

            mlflow.log_params(grid_search.cv_results_['params'][i])
            mlflow.log_metric("accuracy", grid_search.cv_results_['mean_test_score'][i])
    
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy", best_score)

    train_df = X_train.copy()
    train_df['Outcome'] = y_train

    test_df = X_test.copy()
    test_df['Outcome'] = y_test

    train_df = mlflow.data.from_pandas(train_df)
    mlflow.log_input(train_df, "training")
    test_df = mlflow.data.from_pandas(test_df)
    mlflow.log_input(test_df, "validation")
    signature = mlflow.models.infer_signature(X_train, grid_search.best_estimator_.predict(X_train))

    mlflow.log_artifact(__file__)
    mlflow.sklearn.log_model(grid_search.best_estimator_, 'random_forest', signature=signature)
    
    mlflow.set_tag("author","Aryan")
    print(best_params)
    print(best_score)

