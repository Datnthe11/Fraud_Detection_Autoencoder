from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model
import os
import sys

model = load_model("app/best_autoencoder.keras")

# Tạo app FastAPI
app = FastAPI()
class_name = ["Normal","Fraud"]
# Định nghĩa kiểu dữ liệu đầu vào
class InputData(BaseModel):
    features: list  

@app.get("/")
async def root():
    return {"message": "Fraud detection model deployment "}
@app.post("/predict")
async def predict(input: InputData):
    try:
        threshold = 2.3
        input_array = np.array(input.features).reshape(1, -1)

        # Dự đoán
        prediction = model.predict(input_array)
        mse = np.mean(np.power(input_array - prediction,2), axis=1)
        y_pred = 1 if mse > threshold else 0 
        predicted_class = class_name[y_pred]
        return {"prediction": predicted_class}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
