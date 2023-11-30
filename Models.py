import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.applications import Xception



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

    #VGG19 reinziehen
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

def densenet201(train_generator,validation_generator,weight):
    
    input_layer = layers.Input(shape=(224,224,3))

    model_densenet201 = DenseNet201(weights='imagenet',input_tensor = input_layer,include_top = False)

    flatten = tf.keras.layers.Flatten()
    classifier = tf.keras.layers.Dense(1,activation='sigmoid')

    model = tf.keras.models.Sequential([
        model_densenet201,
        flatten,
        classifier
    ])

    #Layer untrainable machen
    for layer in model.layers[:-2]: #auf -1 ändern, wenn nur der finale classifier und keine Dense schicht
        layer.trainable=False

    model_densenet201.summary()
    model.summary()

    model_compiler(model)
    model_fitter(model,train_generator,validation_generator,weight)

    #Confusion matrix anzeigen
    conf_matrix,correct_value_perc = boolean_conf_matrix(model,validation_generator)
    print(conf_matrix)
    print(correct_value_perc)


def InceptionV3(train_generator,validation_generator,weight):
    
    input_layer = layers.Input(shape=(224,224,3))

    model_InceptionV3 = InceptionV3(weights='imagenet',input_tensor = input_layer,include_top = False)

    flatten = tf.keras.layers.Flatten()
    classifier = tf.keras.layers.Dense(1,activation='sigmoid')

    model = tf.keras.models.Sequential([
        model_InceptionV3,
        flatten,
        classifier
    ])

    #Layer untrainable machen
    for layer in model.layers[:-2]: #auf -1 ändern, wenn nur der finale classifier und keine Dense schicht
        layer.trainable=False

    model_InceptionV3.summary()
    model.summary()

    model_compiler(model)
    model_fitter(model,train_generator,validation_generator,weight)

    #Confusion matrix anzeigen
    conf_matrix,correct_value_perc = boolean_conf_matrix(model,validation_generator)
    print(conf_matrix)
    print(correct_value_perc)

def ResNet152V2(train_generator,validation_generator,weight):
    
    input_layer = layers.Input(shape=(224,224,3))

    model_ResNet152V2 = ResNet152V2(weights='imagenet',input_tensor = input_layer,include_top = False)

    flatten = tf.keras.layers.Flatten()
    classifier = tf.keras.layers.Dense(1,activation='sigmoid')

    model = tf.keras.models.Sequential([
        model_ResNet152V2,
        flatten,
        classifier
    ])

    #Layer untrainable machen
    for layer in model.layers[:-2]: #auf -1 ändern, wenn nur der finale classifier und keine Dense schicht
        layer.trainable=False

    model_ResNet152V2.summary()
    model.summary()

    model_compiler(model)
    model_fitter(model,train_generator,validation_generator,weight)

    #Confusion matrix anzeigen
    conf_matrix,correct_value_perc = boolean_conf_matrix(model,validation_generator)
    print(conf_matrix)
    print(correct_value_perc)

def Xception(train_generator,validation_generator,weight):
    
    input_layer = layers.Input(shape=(224,224,3))

    model_Xception = Xception(weights='imagenet',input_tensor = input_layer,include_top = False)

    flatten = tf.keras.layers.Flatten()
    classifier = tf.keras.layers.Dense(1,activation='sigmoid')

    model = tf.keras.models.Sequential([
        model_Xception,
        flatten,
        classifier
    ])

    #Layer untrainable machen
    for layer in model.layers[:-2]: #auf -1 ändern, wenn nur der finale classifier und keine Dense schicht
        layer.trainable=False

    model_Xception.summary()
    model.summary()

    model_compiler(model)
    model_fitter(model,train_generator,validation_generator,weight)

    #Confusion matrix anzeigen
    conf_matrix,correct_value_perc = boolean_conf_matrix(model,validation_generator)
    print(conf_matrix)
    print(correct_value_perc)


def model_compiler(model):
    model.compile(optimizer='Adam',
            loss='binary_crossentropy', #evtl binary_focal_crossentropy
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