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


#Globally create Keras callbacks

#Save weights and model from best epoch
checkpoint_path = os.path.join(os.getcwd(),'bestmodel.h5')
model_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_path,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1
)

#Create early stopper to prevent overfitting
stopper = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss'
    ,mode='min'
    ,verbose=1
    ,patience=3
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
    
    input_layer = layers.Input(shape=(373,373,3))

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
    
    input_layer = layers.Input(shape=(373,373,3))

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
    
    input_layer = layers.Input(shape=(373,373,3))

    model_Xception = tf.keras.applications.Xception(weights='imagenet',input_tensor = input_layer,include_top = False)

    flatten = tf.keras.layers.Flatten()
    classifier = tf.keras.layers.Dense(1,activation='sigmoid')
    globalaverage = tf.keras.layers.GlobalAveragePooling2D()
    dense1 = tf.keras.layers.Dense(128,activation='relu')
    dropout = tf.keras.layers.Dropout(0.5)

    model = tf.keras.models.Sequential([
        model_Xception,
        flatten,
        #globalaverage,
        #dense1,
        #dropout,
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


def model_compiler(model):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=['accuracy','FalseNegatives','FalsePositives','Precision','Recall'])
    

def model_fitter(model,train_generator,validation_generator,weight):
    history = model.fit(train_generator,validation_data=validation_generator,epochs=20
                        ,class_weight = {0: 1, 1: weight}
                        ,callbacks =[model_callback,stopper]
                        #,steps_per_epoch=500
                        )

    #Create plots
    plt.figure(figsize=(12, 6))

    #Plot training and validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    #Plot training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Show the plots
    plt.show()


def model_evaluater(test_generator):
    #Load model from best epoch
    model = tf.keras.models.load_model(os.path.join(os.getcwd(),"bestmodel.h5"))

    #Evaluate the model and save to results
    results = model.evaluate(test_generator)

    #Create dict with evaluation metrics for testing dataset
    eval_metrics = {"Accuracy: ": results[1]
                    ,"Loss: ": results[0]
                    ,"False_Negatives: ":int(results[2])
                    ,"False_Positives: ":int(results[3])
                    ,"Precision: ":results[4]
                    ,"Recall: ":results[5]
                    }
    print(eval_metrics)


    False_Negatives = int(eval_metrics['False_Negatives: '])
    False_Positives = int(eval_metrics['False_Positives: '])

    True_Positives = int((eval_metrics['Precision: ']*False_Positives)/(1-eval_metrics['Precision: ']))
    True_Negatives = int((test_generator.n)-True_Positives-False_Negatives-False_Positives)

    Precicion = eval_metrics['Precision: ']
    Recall = eval_metrics['Recall: ']
    Accuracy = eval_metrics['Accuracy: ']


    confusion_matrix = np.array([[True_Positives,False_Positives],[False_Negatives,True_Negatives]])

    print('Confusion Matrix:')
    print(confusion_matrix)

    print('Accuracy:')
    print(Accuracy)

    print('Precicion:')
    print(Precicion)

    print('Recall:')
    print(Recall)

    
    