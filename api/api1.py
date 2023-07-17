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
from keras.preprocessing import image

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

#Hey sarika, here i get the img from the front end via post api. The incoming img will be of varying size.
#since i trainrd the images in 100,100 dimension, the incoming img should also be in this format.
def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data))

    #Here I tried resizing but didnt work
    #print(image)
    #image = image.resize(100, 100)
    #print(image)
    #

    img = np.array(image)
    return img 
    
MODEL = tf.keras.models.load_model("./Model/Initial_Model_89")
CLASS_NAMES = ["Butterfly", "Dragonfly", "Grasshopper", "Ladybird", "Mosquito"]

@app.get("/ping")
async def ping():
    return "Hello, Naresh server is running"

@app.post("predict/")
async def predict(file: UploadFile=File(...)):
    image = read_file_as_image(await file.read())
    #print(image.shape)
    #print(image)
    test_image = image.load_img(image, target_size = (100, 100))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    prediction = MODEL.predict(test_image)
    #image_batch = np.expand_dims(image, 0) #axis = 0 is row level high dim
    #model accepts images as batch as its trained by that way
    #prediction = MODEL.predict(image_batch)
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
