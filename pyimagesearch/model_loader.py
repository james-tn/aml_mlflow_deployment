from pyimagesearch import config
import mlflow.pyfunc
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import imutils
import pickle
import sklearn
import cv2
import numpy as np
import os
import base64
from io import BytesIO
from PIL import Image

def base64ToImg(base64ImgString):

    if base64ImgString.startswith('b\''):

        base64ImgString = base64ImgString[2:-1]

    base64Img   =  base64ImgString.encode('utf-8')

    decoded_img = base64.b64decode(base64Img)

    img_buffer  = BytesIO(decoded_img)

    img = Image.open(img_buffer)
    return img

class Object_Detection(mlflow.pyfunc.PythonModel):

    def __init__(self, path):
        lb_path = os.path.join(path,config.LB_PATH)
        model_path = os.path.join(path,config.MODEL_PATH)
        
        self.lb = pickle.loads(open(lb_path, "rb").read())
        self.model = load_model(model_path)

    def predict(self, imagePathDF):
        
        result=[]
        
        for row_idx in range(imagePathDF.shape[0]):
            base64_string = imagePathDF[0][row_idx]
            img = base64ToImg(base64_string)
            image= img.resize((224, 224))
            image = img_to_array(image) / 255.0
            image = np.expand_dims(image, axis=0)

            # predict the bounding box of the object along with the class
            # label
            (boxPreds, labelPreds) = self.model.predict(image)
            (startX, startY, endX, endY) = boxPreds[0]

            # determine the class label with the largest predicted
            # probability
            i = np.argmax(labelPreds, axis=1)
            label = self.lb.classes_[i][0]

            result.append((float(startX), float(startY), float(endX), float(endY), label))
        return result
def _load_pyfunc(path):
    return Object_Detection(path)
    

