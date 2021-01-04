import json
import numpy as np
import os
from sklearn.externals import joblib


def init():
    global model
    # if model to be deployed is from automl run
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'best_automl_model.pkl')
    # if model to be deployed is from hyperdrive run
    # model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'best_hyperdrive_model.pkl')
    model = joblib.load(model_path)

def run(data):
    try:
        data = np.array(json.loads(data))
        result = model.predict(data)
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error