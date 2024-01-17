import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os


name = 'IMG0002515_flipped.jpg'


os.chdir('C:/Users/paulz/Documents/ProgrammingDigitalProjects_v2/models')

model = load_model('xception_90,21.h5')

img_path = 'C:/Users/paulz/Documents/ProgrammingDigitalProjects_v2/Dataset_FracAtlas/images/full_augmented/' + name

img = tf.keras.preprocessing.image.load_img(img_path,target_size=(373,373))

img_array = tf.keras.preprocessing.image.img_to_array(img)

img_array = np.expand_dims(img_array,axis=0)
img_array/=255.0


predictions = model.predict(img_array)

print(predictions)