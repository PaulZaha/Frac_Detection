import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import VGG19

import matplotlib.pyplot as PLT


from Data_import import *



def vgg16(train_generator,validation_generator,weight):
    #Set input layer
    input_layer = layers.Input(shape=(224,224,3))

    #VGG16 reinziehen
    model_vgg16=VGG16(weights='imagenet',input_tensor=input_layer,include_top=False)

    #Flatten und Classifier hinzufügen
    flatten=tf.keras.layers.Flatten()
    classifier = tf.keras.layers.Dense(1,activation='softmax')

    #Model zusammenfügen
    model = tf.keras.models.Sequential([
        model_vgg16,
        flatten,
        classifier
    ])

    #Layer untrainable machen
    for layer in model.layers[:-1]:
        layer.trainable=False

    model_vgg16.summary()
    model.summary()

    model_compiler(model)
    model_fitter(model,train_generator,validation_generator,weight)

    #Confusion matrix anzeigen
    conf_matrix,correct_value_perc = boolean_conf_matrix(model,validation_generator)
    print(conf_matrix)
    print(correct_value_perc)

def vgg19(train_generator,validation_generator,weight):
    #Set input layer
    input_layer = layers.Input(shape=(224,224,3))

    #VGG16 reinziehen
    model_vgg19=VGG19(weights='imagenet',input_tensor=input_layer,include_top=False)

    #Flatten und Classifier hinzufügen
    flatten=tf.keras.layers.Flatten()
    dense1=tf.keras.layers.Dense(512,activation='relu')
    classifier = tf.keras.layers.Dense(1,activation='sigmoid')

    #Model zusammenfügen
    model = tf.keras.models.Sequential([
        model_vgg19,
        flatten,
        #dense1,
        classifier
    ])

    #Layer untrainable machen
    for layer in model.layers[:-2]: #auf -1 ändern, wenn nur der finale classifier und keine Dense schicht
        layer.trainable=False

    model_vgg19.summary()
    model.summary()

    model_compiler(model)
    model_fitter(model,train_generator,validation_generator,weight)

    #Confusion matrix anzeigen
    conf_matrix,correct_value_perc = boolean_conf_matrix(model,validation_generator)
    print(conf_matrix)
    print(correct_value_perc)

def model_CNN(train_generator,validation_generator,weight):
    model = tf.keras.models.Sequential([
    tf.keras.layers.Resizing(373,373),

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
    model.compile(optimizer='Adam',
            loss='binary_crossentropy',
            metrics=['BinaryAccuracy'])
    
def model_fitter(model,train_generator,validation_generator,weight):
    history = model.fit(train_generator,validation_data=validation_generator,epochs=10
                        ,class_weight = {0: 1, 1: weight}
                        )


    #Plotting accuracy and loss over epoch time
    plt.plot(history.history['binary_accuracy'], label = 'accuracy')
    plt.plot(history.history['loss'], label = 'loss')
    plt.plot(history.history['val_binary_accuracy'],label='validation accuracy')
    plt.plot(history.history['val_loss'], label = 'validation loss')
    plt.legend()
    plt.show()


def boolean_conf_matrix(model,generator):
    true_labels = np.array(generator.classes)
    print(true_labels)
    predicted_labels_binary = (model.predict(generator)[:, 0] > 0.5).astype(int)
    print(predicted_labels_binary)
    conf_matrix = tf.math.confusion_matrix(true_labels,predicted_labels_binary)

    correct_value_perc = ((conf_matrix[0,0] + conf_matrix[1,1]) / np.sum(conf_matrix))

    return conf_matrix,correct_value_perc