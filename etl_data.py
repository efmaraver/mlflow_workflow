import tempfile
import os
import pandas as pd
import numpy as np
import mlflow



def etl_data(bankloan_csv):
    with mlflow.start_run() as mlrun:
        tmpdir = tempfile.mkdtemp()
        bankloan_clean_dir = os.path.join(tmpdir, 'Bank_Loan_clean')

        data = pd.read_csv(bankloan_csv)
        # No columns have null data in the file
        data.apply(lambda x : sum(x.isnull()))

        # Eye balling the data
        data.describe().transpose()

        # finding unique data
        data.apply(lambda x: len(x.unique()))

        # there are 52 records with negative experience. Before proceeding any further we need to clean the same
        data[data['Experience'] < 0]['Experience'].count()

        #clean the negative variable
        dfExp = data.loc[data['Experience'] >0]
        negExp = data.Experience < 0
        column_name = 'Experience'
        mylist = data.loc[negExp]['ID'].tolist() # getting the customer ID who has negative experience

        for id in mylist:
            age = data.loc[np.where(data['ID']==id)]["Age"].tolist()[0]
            education = data.loc[np.where(data['ID']==id)]["Education"].tolist()[0]
            df_filtered = dfExp[(dfExp.Age == age) & (dfExp.Education == education)]
            exp = df_filtered['Experience'].median()
            data.loc[data.loc[np.where(data['ID']==id)].index, 'Experience'] = exp
        # checking if there are records with negative experience
        data[data['Experience'] < 0]['Experience'].count()


        data.to_csv(os.path.join(bankloan_clean_dir,'Bank_Loan_clean.csv'))

        ratings_df.write.parquet(ratings_parquet_dir)
        print("Uploading Parquet ratings: %s" % bankloan_clean_dir)
        mlflow.log_artifacts(bankloan_clean_dir, "bankloan_clean_dir")


if __name__ == '__main__':
    etl_data()
