import tensorflow as tf
from tensorflow.keras.models import load_model

import pandas as pd
import numpy as np
from fastapi import FastAPI
from uvicorn import run
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

input_data_repo = "./inputs/"
models_repo = "./models/"

# define a custom optimization metric
class R2Metric(tf.keras.metrics.Metric):
    def __init__(self, name='r_squared', **kwargs):
        super(R2Metric, self).__init__(name=name, **kwargs)
        self.sse = self.add_weight(name='sse', initializer='zeros')
        self.sst = self.add_weight(name='sst', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Residual sum of squares (sse)
        sse = tf.reduce_sum(tf.square(y_true - y_pred))

        # Total sum of squares (sst)
        sst = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))

        # Update the states
        self.sse.assign_add(sse)
        self.sst.assign_add(sst)

    def result(self):
        return 1 - (self.sse / (self.sst + tf.keras.backend.epsilon()))  # To prevent division by zero

    def reset_states(self):
        self.sse.assign(0.0)
        self.sst.assign(0.0)

    def result(self):
        return 1 - (self.sse / (self.sst + tf.keras.backend.epsilon()))  # To prevent division by zero

    def reset_states(self):
        self.sse.assign(0.0)
        self.sst.assign(0.0)

# input features
input_features = {} 
with open(input_data_repo + 'features.json', 'r') as file:
    input_features = json.load(file)

# Load the saved model
loaded_model = load_model(models_repo + "main/leap_sim_tf_model_2024-10-17 14:39_Loss_3.904_rootSquaredError_0.9943.keras", custom_objects={'R2Metric': R2Metric})  # Include custom metric

# Create a FastAPI app
app = FastAPI()

pca_models = dict()
for feature_name in input_features.keys():
    logging.info(f"Loading pcs of composite feature {feature_name}")
    pca_models[feature_name] = joblib.load(models_repo + 'pcs/composit_feature_' + feature_name + '_pca_model2.pkl')

# Define a function to scale input data
scaling_models = dict()
def scale_input_data(feature_name, input_data_df):
    from sklearn.preprocessing import StandardScaler
    # reading the scaling model
    logging.info(f"Loading scaling model of composit feature {feature_name}")
    scaling_models[feature_name] = joblib.load(models_repo + 'scaling/composit_feature_' + feature_name + '_scaling_model2.pkl')

    # transform the inuput data
    logging.debug(f"Input data before scaling: {input_data_df}")
    scaled_data = scaling_models[feature_name].transform(input_data_df)
    logging.debug(f"Input data after scaling: {scaled_data}")
    return scaled_data


# Define a function to apply PCA
def apply_scaling_pca(input_data_df):
    # use the pca_models dict to get the pca model for the feature
    # apply the pca model to the feature
    # the pcas of the features are passed down as input to the model

    # Initialize an empty list to store the principal component representation for the input data
    the_principal_component = []
    for feature_name in input_features.keys():
        scaled_feature_ndarray = scale_input_data (feature_name, input_data_df[input_features[feature_name]])
        principal_components = pca_models[feature_name].transform(input_data_df[input_features[feature_name]])

        logging.debug(f"Principal components of feature {feature_name} are {principal_components}")
        the_principal_component.append(principal_components)

    # return a list of input pcs required as input to the model
    return the_principal_component

# Define a prediction endpoint
@app.post("/predict/") 
async def predict(input_data: dict):
    """Makes predictions using the exported TensorFlow model.

    Args:
        input_data (dict): A dictionary containing the input features
                           for the prediction. It should have the same
                           structure as the data used for training.

    Returns:
        dict: A dictionary containing the model's predictions.
    """

    # Create a Pandas DataFrame from the input data
    input_data_df = pd.DataFrame([input_data])
    logging.info(f"Payload received")

    # Convert the scaled data to principal components
    principal_components = apply_scaling_pca(input_data_df)
    logging.info(f"PCs constructed from input data")

    # Make the prediction
    prediction = loaded_model.predict(principal_components)

    # Return the prediction
    return {"prediction": prediction.tolist()}

if __name__ == "__main__":
    # Run the FastAPI app using Uvicorn
    run(app, host="0.0.0.0", port=8000)
