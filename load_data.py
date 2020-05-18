import pandas as pd
import numpy as np
import xlrd
import mlflow
import tempfile

def load_raw_data(url):
    with mlflow.start_run() as mlrun:
        xl = pd.ExcelFile('Bank_Personal_Loan_Modelling.xlsx')
        data = xl.parse('Data')

        local_dir = tempfile.mkdtemp()
        extracted_dir = os.path.join(local_dir, 'Bank_Loan')
        extracted_file = os.path.join(extracted_dir, 'Bank_Loan.csv')
        data.to_csv(extracted_file)
        mlflow.log_artifact(extracted_file, "bankloan-csv-dir")

if __name__ == '__main__':
    load_raw_data()
