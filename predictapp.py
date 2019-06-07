'''from PIL import Image
import numpy as np
import os
from random import shuffle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np'''

#-------- serving_sample_request.py ---------#
import base64
import numpy as np
import io
#import PIL import Image
from PIL import Image
#from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow import keras
#import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
#from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from flask import Flask, redirect
from flask import request
from flask import jsonify
#--------------------------------------------------#

app = Flask(__name__)
tf.enable_eager_execution() # Disble graph Tensorflow

def get_model():
    #---- Load RUN Model (.h5) ---#
    global model,graph
    model = load_model('modelDogCat.h5')
    #model.load_weights('modelDogCat.h5')
    #model._make_predict_function()
    print(" * Model loaded !")

 
def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size,Image.ANTIALIAS)
    image = img_to_array(image)
    image = np.expand_dims(image, axis = 0)
    print('preprocess_image: ' + str(image))
    return image



# Evaluate the restored model.
print("* Loading Model...")
get_model()
@app.route("/predict", methods=["POST"])
def predict():
    #global graph
    #graph = tf.Graph()
    #with graph.as_default():
    message = request.get_json(force=True)
    encoded = message['image']
    #print("encoded: " + str(encoded))
    #try:
    print("Hello Pon !!!")
    decoded = base64.b64decode(encoded)
    #print("decoded: " + str(decoded))

    image = Image.open(io.BytesIO(decoded))
    print("image Open: " + str(image))
    processed_image = preprocess_image(image, target_size=(64,64))
    processed_image = np.float32(processed_image)
    print('processed_image: OK!!')
 
    try:
        model.summary()
            
        print('processed_image: --->  ' + str(processed_image))
        # print(model(processed_image))
        prediction = model.predict(processed_image).tolist()
        print('prediction:>> ' + str(prediction))
        d = {0:'cat', 1:'dog'}
        d[prediction[0][0]]
        print(d[prediction[0][0]])
         
        response = {
            'prediction' : {
                #'dog' : prediction[0][1],
                #'cat' : d[prediction[0][0]]
                'result' : d[prediction[0][0]]
             }
        }
            
            #return jsonify(prediction)
        return jsonify(response)
    except Exception as e:
        print(e)

@app.route("/")
def hello():
    return "Hello World! Flask Pon Test"
       
if __name__ == "__main__":
    # Only for debugging while developing#
    #app.run(debug=True)
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
