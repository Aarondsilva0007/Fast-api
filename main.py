from tensorflow.keras.models import load_model
from typing import Optional
import numpy as np
import json
from fastapi import FastAPI
import pickle as pikl
from keras import backend as K

app = FastAPI()

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 
from tensorflow.keras.models import load_model
loaded_model = load_model("er_model0", custom_objects={'root_mean_squared_error': root_mean_squared_error})


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/predict")
def scale_predict(text: str):
    # scaler = pickle.load(open("scaler.pkl", 'rb'))
    scaler1 = pikl.load(open("scalerlab.pkl", 'rb'))
    d = json.loads(text)
    numpy_2d_arrays = np.array(d)
    # sh = numpy_2d_arrays.shape
    # arr = np.reshape(numpy_2d_arrays,(1,numpy_2d_arrays.shape[0], numpy_2d_arrays.shape[1]))
    res = loaded_model.predict(numpy_2d_arrays)
    # return {"result_ns":f'{res}', "result_s":f'{scaler1.inverse_transform(res)}'}
    return {"pred": f'{scaler1.inverse_transform(res)[0][0]}', "type": numpy_2d_arrays.shape}


@app.get('/Welcome')
def get_name(name: str):
    return {'Welcome To Krish Youtube Channel': f'{name}'}
