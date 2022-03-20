from tensorflow.keras.models import load_model
from typing import Optional
import numpy as np
import json
from fastapi import FastAPI
import pickle as pikl
from keras import backend as K
import uvicorn

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = FastAPI()


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


loaded_model = load_model("D:\Sem7\BE_Project\machine_learning\\final_model", custom_objects={
                          'root_mean_squared_error': root_mean_squared_error})


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/predict")
def scale_predict(text: str):
    scaler = pikl.load(open("D:\Sem7\BE_Project\machine_learning\scalerfinal.pkl", 'rb'))
    scaler1 = pikl.load(open("D:\Sem7\BE_Project\machine_learning\scalerlabfinal.pkl", 'rb'))
    # print(text)
    d = json.loads(text)
    # d.reshape(1,5,5)
    d = scaler.transform(d)
    print(d)
    numpy_2d_arrays = np.array(d, dtype=object)
    print(numpy_2d_arrays.shape)
    print(numpy_2d_arrays)
    d=d.reshape(1,5,5)
    # print(numpy_2d_arrays)
    # print(numpy_2d_arrays.shape)
    # arr = np.reshape(numpy_2d_arrays,(1,numpy_2d_arrays.shape[0], numpy_2d_arrays.shape[1]))
    # print(type(arr))
    res = loaded_model.predict(d)
    print(res)
    # return {"result_ns":f'{res}', "result_s":f'{scaler1.inverse_transform(res)}'}
    return {"pred": f'{scaler1.inverse_transform(res)[0][0]}'}

@app.get('/Welcome')
def get_name(name: str):
    return {'Welcome To Krish Youtube Channel': f'{name}'}


if __name__ == '__main__':
    uvicorn.run(app,host='127.0.0.1', port=8000, debug=True)




# [[0.33333333, 0.7, 0.33333333, 0.03793884, 0.75],[0, 0.3333333, 0.03510759, 0.5],[0.33333333, 0.7, 0.33333333, 0.07644394, 0.75], [0.33333333, 0.7, 0.33333333, 0.51189128, 0.875], [0.33333333, 0.7 , 0.33333333, 0.03114383, 0.5]]
# [[26.0,67.00, 27.0, 188, 274],[26.0, 67.00, 26.5, 67, 280],[26.0, 67.00, 26.0, 62, 276],[26.0, 67.00, 26.5, 135, 280],[26.0, 67.00, 26.5, 904, 282]]
