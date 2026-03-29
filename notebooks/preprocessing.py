import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config_utils import RAW_DATA_PATH, PROCESSED_DATA_PATH
from src.data_utils import load_data, save_processed_data

def _handle_outliers(data, outliers):
    for col in outliers:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
    save_processed_data(data, PROCESSED_DATA_PATH)
    print('New data is stored in: ', PROCESSED_DATA_PATH)



def preprocess():
    # Load the dataset
    data = load_data(RAW_DATA_PATH)
    data.drop(['Unnamed: 0'], axis=1, inplace=True)
    print('Data: ',data.head())
    print('Data info: ',data.info())
    print('Data description: ',data.describe())

    # data cleaning
    print('Data shape: ',data.shape)
    duplicated = data.duplicated().sum()
    print('Duplicated: ', duplicated)
    if duplicated > 0:
        data.drop_duplicates(inplace=True)
    print('Data shape a fter removing duplicates: ', data.shape)
    missing_values = data.isnull().sum().sum()
    print('Missing values: ', missing_values)
    if missing_values > 0.3 * len(data):
        data.fillna(value=data.mean(),inplace=True)
    else:
        data.dropna(inplace=True)
    print('Data shape after removing missing values: ', data.shape)
    _handle_outliers(data, ['price', 'duration'])
    print('Data shape after removing outliers values: ', data.shape)

if __name__ == "__main__":
    preprocess()
    print("Preprocessing completed.")

