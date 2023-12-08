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

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as PLT

from Data_import import *

checkpoint_path = os.path.join(os.getcwd(),'bestmodel.h5')
model_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_path,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1
)


def vgg16(train_generator,validation_generator,test_generator,weight):
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

    model_compiler(model)
    print("Ab hier: Model Fitting")
    model_fitter(model,train_generator,validation_generator,weight)
    print("Ab hier: Model evaluation")
    model_evaluater(test_generator)

def vgg19(train_generator,validation_generator,test_generator,weight):
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

    model_compiler(model)
    print("Ab hier: Model Fitting")
    model_fitter(model,train_generator,validation_generator,weight)
    print("Ab hier: Model evaluation")
    model_evaluater(test_generator)

def densenet201(train_generator,validation_generator,test_generator,weight):
    
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
    for layer in model.layers[:-1]: #auf -1 ändern, wenn nur der finale classifier und keine Dense schicht
        layer.trainable=False


    model_compiler(model)
    print("Ab hier: Model Fitting")
    model_fitter(model,train_generator,validation_generator,weight)
    print("Ab hier: Model evaluation")
    model_evaluater(test_generator)

def InceptionV3(train_generator,validation_generator,test_generator,weight):
    
    input_layer = layers.Input(shape=(224,224,3))

    model_InceptionV3 = tf.keras.applications.InceptionV3(input_tensor = input_layer,include_top = False,weights='imagenet')

    flatten = tf.keras.layers.Flatten()
    classifier = tf.keras.layers.Dense(1,activation='sigmoid')

    model = tf.keras.models.Sequential([
        model_InceptionV3,
        flatten,
        classifier
    ])

    #Layer untrainable machen
    for layer in model.layers[:-1]: #auf -1 ändern, wenn nur der finale classifier und keine Dense schicht
        layer.trainable=False

    model_compiler(model)
    print("Ab hier: Model Fitting")
    model_fitter(model,train_generator,validation_generator,weight)
    print("Ab hier: Model evaluation")
    model_evaluater(test_generator)

def ResNet152V2(train_generator,validation_generator,test_generator,weight):
    
    input_layer = layers.Input(shape=(224,224,3))

    model_ResNet152V2 = tf.keras.applications.ResNet152V2(weights='imagenet',input_tensor = input_layer,include_top = False)

    flatten = tf.keras.layers.Flatten()
    classifier = tf.keras.layers.Dense(1,activation='sigmoid')

    model = tf.keras.models.Sequential([
        model_ResNet152V2,
        flatten,
        classifier
    ])

    #Layer untrainable machen
    for layer in model.layers[:-1]: #auf -1 ändern, wenn nur der finale classifier und keine Dense schicht
        layer.trainable=False

    model_compiler(model)
    print("Ab hier: Model Fitting")
    model_fitter(model,train_generator,validation_generator,weight)
    print("Ab hier: Model evaluation")
    model_evaluater(test_generator)

def Xception(train_generator,validation_generator,test_generator,weight):
    
    input_layer = layers.Input(shape=(224,224,3))

    model_Xception = tf.keras.applications.Xception(weights='imagenet',input_tensor = input_layer,include_top = False)

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

    model_compiler(model)
    print("Ab hier: Model Fitting")
    model_fitter(model,train_generator,validation_generator,weight)
    print("Ab hier: Model evaluation")
    model_evaluater(test_generator)


def model_compiler(model):
    model.compile(optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.BinaryCrossentropy(), #evtl binary_focal_crossentropy
            metrics=['accuracy'])
    

def model_fitter(model,train_generator,validation_generator,weight):
    history = model.fit(train_generator,validation_data=validation_generator,epochs=10
                        ,class_weight = {0: 1, 1: weight}
                        ,callbacks =[model_callback]
                        #,steps_per_epoch=500
                        )
    #Plotting accuracy and loss over epoch time
    plt.plot(history.history['accuracy'], label = 'accuracy')
    plt.plot(history.history['loss'], label = 'loss')
    #plt.plot(history.history['val_accuracy'],label='validation accuracy')
    #plt.plot(history.history['val_loss'], label = 'validation loss')
    plt.legend()
    plt.show()



def model_evaluater(test_dataset):
    model = tf.keras.models.load_model(os.path.join(os.getcwd(),"bestmodel.h5"))

    #results = model.evaluate(test_dataset)

    true_labels = np.array(test_dataset.classes)
    predicted_labels = (model.predict(test_dataset)[:, 0] > 0.5).astype(int)

    #Confusion Matrix
    conf_matrix = tf.math.confusion_matrix(true_labels,predicted_labels)
    conf_matrix = tf.reverse(conf_matrix, axis=[0])
    conf_matrix = tf.reverse(conf_matrix, axis=[1])

    TP = conf_matrix[1, 1]
    TN = conf_matrix[0, 0]
    FP = conf_matrix[0, 1]
    FN = conf_matrix[1, 0]


    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    print("Confusion Matrix:")
    print(conf_matrix)
    print("Sensitivity (True Positive Rate):", sensitivity)
    print("Specificity (True Negative Rate):", specificity)
    print("Accuracy:", accuracy)