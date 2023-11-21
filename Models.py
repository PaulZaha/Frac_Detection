import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from Data_import import *



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
    tf.keras.layers.Resizing(100,100),

    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    #tf.keras.layers.RandomRotation(factor=(-0.2,0.2),fill_mode="reflect"),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    #tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    #tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    #tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    #tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model_compiler(model)
    #model.summary()
    model_fitter(model,train_generator,weight)
    model_evaluater(model,validation_generator)

    

    conf_matrix,correct_value_perc = boolean_conf_matrix(model,validation_generator)
    print(conf_matrix)
    print(correct_value_perc)



    

def model_compiler(model):
    model.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'])
    
def model_fitter(model,train_generator,weight):
    history = model.fit(train_generator,epochs=3,class_weight = {0: 1, 1: weight})
    print(history.history.keys())

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['loss'])
    plt.show()

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