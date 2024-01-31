import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os


name = 'IMG0000143_zoomed_flipped.jpg'


os.chdir(os.path.join(os.getcwd()+'/Ergebnisse','EfficientNetB4','optimization2'))

model = load_model('bestmodel.h5')
os.chdir(os.path.join(os.getcwd(),'..','..','..'))
path = "C:/Users/paulz/OneDrive/Dokumente/ProgrammingDigitalProjects_v2/Dataset_FracAtlas/images/edge_detection/" + name

img = tf.keras.preprocessing.image.load_img(path,target_size=(380,380))

img_array = tf.keras.preprocessing.image.img_to_array(img)

img_array = np.expand_dims(img_array,axis=0)
#img_array/=255.0


predictions = model.predict(img_array)

print(predictions)