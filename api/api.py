# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 14:03:23 2022

@author: nary2
"""

from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from skimage.transform import resize
from fastapi.middleware.cors import CORSMiddleware
import cv2 as cv

app = FastAPI()

#CORS 
origins = [
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

def read_file_as_image(data):
    #size = 100, 100
    image = Image.open(BytesIO(data))
    #image.thumbnail(size,Image.ANTIALIAS)
    #print(image)
    img = image.resize((100, 100))
    img = np.array(img, dtype="float32")
    img = img.reshape(100, 100, 3)
    #Convert the incoming data to 100*100
    #img = resize(img, (100, 100))
    return img 
    
MODEL = tf.keras.models.load_model("./Model/Initial_Model_89")
CLASS_NAMES = ["Butterfly", "Dragonfly", "Grasshopper", "Ladybird", "Mosquito"]

@app.get("/ping")
async def ping():
    return "Hello, Naresh server is running"

@app.post("/predict")
async def predict(file: UploadFile=File(...)):
    #image = cv.resize(image, (100,100),interpolation = cv.INTER_AREA)
    image = read_file_as_image(await file.read())
    #print(image.shape)
    #print(image)
    image_batch = np.expand_dims(image, 0) #axis = 0 is row level high dim
    #model accepts images as batch as its trained by that way
    prediction = MODEL.predict(image_batch)
    #print(prediction)
    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    print(predicted_class)
    print(confidence)
    return {
        "prediction": predicted_class,
        "confidence": float(confidence)
    }
    


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=5500)