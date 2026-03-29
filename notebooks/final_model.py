import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config_utils import PROCESSED_DATA_PATH, MODEL_PATH
from src.model_utils import train_model, save_model
from src.data_utils import load_data
from sklearn.preprocessing import LabelEncoder

# 98.99% accuracy
data = load_data(PROCESSED_DATA_PATH)
print("Data: ", data.head())
print('Info: '  , data.info())
print('Describe: '  , data.describe())
print('Shape: '  , data.shape)

# encoding
label_encoder = LabelEncoder()
for column in data.select_dtypes(include=['object']).columns:
    data[column] = label_encoder.fit_transform(data[column])
print("Encoding completed successfully.")

# Split the data into training and testing sets
x = data.drop(columns=['price'])
y = data['price']

# Train the model
model, r2_score = train_model(x, y)
print("Model training completed successfully.")
print("R2 Score: ", r2_score)

# Save the model
save_model(model, MODEL_PATH)
print("Model saved successfully.")