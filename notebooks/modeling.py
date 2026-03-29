import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pycaret.regression import *
from src.config_utils import PROCESSED_DATA_PATH
from src.data_utils import load_data

data = load_data(PROCESSED_DATA_PATH)
print("Data: ", data.head())

reg_setup = setup(data=data.sample(100000), target='price', session_id=123, verbose=False)
print("Setup completed successfully.")

best_model = compare_models()
print("Best regression model selected:", best_model)
tuned_model = tune_model(best_model, fold=5, n_iter=5)
print("Tuning completed successfully.", tuned_model)