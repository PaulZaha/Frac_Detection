import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential



def model_sequential(train_generator,validation_generator):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(224 , 224, 3)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    
    #model.summary()
    model_compiler(model)
    model_fitter(model,train_generator)
    model_evaluater(model,validation_generator)

    conf_matrix,correct_value_perc = boolean_conf_matrix(model,validation_generator)
    print(conf_matrix)
    print(correct_value_perc)
    



def model_CNN(train_generator,validation_generator,weight):
    model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 200x200 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(373, 373, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),





    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model_compiler(model)
    model.summary()
    model_fitter(model,train_generator,weight)
    model_evaluater(model,validation_generator)

    conf_matrix,correct_value_perc = boolean_conf_matrix(model,validation_generator)
    print(conf_matrix)
    print(correct_value_perc)


#CNN funktioniert noch überhaupt nicht. Klassifiziert jedes Bild als 1 (=gebrochen)
def model_CNN_old(train_generator, validation_generator):
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
    ])
    model_compiler(model)
    model.summary()
    model_fitter(model,train_generator)
    model_evaluater(model,validation_generator)

    conf_matrix,correct_value_perc = boolean_conf_matrix(model,validation_generator)
    print(conf_matrix)
    print(correct_value_perc)




def model_compiler(model):
    model.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'])
    
def model_fitter(model,train_generator,weight):
    model.fit(train_generator,epochs=5,class_weight = {0: 1, 1: weight})

def model_evaluater(model,validation_generator):
    model.evaluate(validation_generator,verbose=1)

def boolean_conf_matrix(model,generator):
    true_labels = np.array(generator.classes)
    print(true_labels)
    predicted_labels_binary = (model.predict(generator)[:, 0] > 0.5).astype(int)
    print(predicted_labels_binary)
    conf_matrix = tf.math.confusion_matrix(true_labels,predicted_labels_binary)

    correct_value_perc = ((conf_matrix[0,0] + conf_matrix[1,1]) / np.sum(conf_matrix))

    return conf_matrix,correct_value_perc