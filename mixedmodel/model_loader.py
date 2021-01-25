import mlflow.pyfunc
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
import numpy as np
import ast
import pandas as pd
import os
import base64
from io import BytesIO
from PIL import Image
import joblib

def base64ToImg(base64ImgString):

    if base64ImgString.startswith('b\''):

        base64ImgString = base64ImgString[2:-1]

    base64Img   =  base64ImgString.encode('utf-8')

    decoded_img = base64.b64decode(base64Img)

    img_buffer  = BytesIO(decoded_img)

    img = Image.open(img_buffer)
    return img

class Mixed_Model(mlflow.pyfunc.PythonModel):

    def __init__(self, path):
        MODEL_FILE ='rf_sample.pkl'
        model_path = os.path.join(path,MODEL_FILE)
        self.rf_model = joblib.load(model_path)
        base_model = VGG19(weights='imagenet')
        self.model = Model(base_model.input, outputs=base_model.get_layer("fc2").output)

    def predict(self, imagePathDF):
        
        features_list = []
        addional_feature_list=[]

        for row_idx in range(imagePathDF.shape[0]):
            base64_string = imagePathDF.iloc[row_idx]['image']
            img = base64ToImg(base64_string)
            image = img.resize((224, 224))
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            x = preprocess_input(image)
            features = self.model.predict(x)
            features_list.append(features)
            addional_feature = imagePathDF.iloc[row_idx]["additional_feature"]
            addional_feature_list.append(ast.literal_eval(addional_feature))
        new_features = np.concatenate([pd.DataFrame(addional_feature_list).values, np.concatenate(features_list)],
                                      axis=1)
        output = self.rf_model.predict(new_features)

        return output.tolist()
def _load_pyfunc(path):
    return Mixed_Model(path)
    

