import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
import pickle
import mlflow
import click
import mlflow.xgboost


@click.command(help="Given a CSV file (see load_raw_data), transforms it into Parquet "
                    "in an mlflow artifact called 'bankloan-csv-dir'")
@click.option("--bankloan-data")
@click.option("--namecsvetl", default="Bank_Loan_clean.csv")


def train_xgboost(bankloan_data, namecsvetl):
    # Split train-test
    data = pd.read_csv(os.path.join(bankloan_data, namecsvetl), encoding='utf-8', engine='python')
    #data = pd.read_csv(bankloan_data)
    
    train_set, test_set = train_test_split(data.drop(['Unnamed: 0','ID'], axis=1), test_size=0.8 , random_state=100)
    train_set = train_set.dropna()
    test_set = test_set.dropna()
    y_train = train_set.pop('Personal Loan')
    y_test = test_set.pop('Personal Loan')

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(train_set)
    x_test = scaler.transform(test_set)
    
    mlflow.xgboost.autolog()
    with mlflow.start_run():
        # modelo
        model = XGBClassifier()
        model.fit(x_train, y_train.values)

        pred = model.predict(x_test)
        # calculamos precisión y auc
        confusion_matrix(y_test, pred)

        auc = roc_auc_score(y_test, pred)
        acc = accuracy_score(y_test, pred)

        # Set metrics to track
        mlflow.log_metric("auc", auc)
        mlflow.log_metric("acc", acc)
        mlflow.xgboost.log_model(model, "xgboost-model")

if __name__ == '__main__':
    train_xgboost()
