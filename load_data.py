import pandas as pd
import numpy as np
import xlrd
import mlflow
import tempfile
import os
import click

@click.command(help="Upload dataset and saves it as an mlflow artifact "
                    " called 'bankloan-csv-dir'.")
@click.option("--namecsv", default="BankLoan.csv")

def load_data(namecsv):
    with mlflow.start_run() as mlrun:
        xl = pd.ExcelFile('Bank_Personal_Loan_Modelling.xlsx')
        data = xl.parse('Data')
        local_dir = tempfile.mkdtemp()
        extracted_dir = os.path.join(local_dir, 'Bank_Loan')
        os.mkdir(extracted_dir)
        print(os.listdir(extracted_dir))
        extracted_file = os.path.join(extracted_dir, namecsv)
        
        data.to_csv(extracted_file)
        mlflow.log_artifact(extracted_file, "bankloan-csv-dir")

if __name__ == '__main__':
    load_data()
