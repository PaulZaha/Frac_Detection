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
import tensorboard
import matplotlib.pyplot as PLT
from datetime import datetime



from pipelines import *


#TensorBoard Callback
log_dir = "logs" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


#Callback, to save weights and model from best epoch
checkpoint_path = os.path.join(os.getcwd(),'bestmodel.h5')
model_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_path,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1
)

#Callback: Create early stopper to prevent overfitting
stopper = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss'
    ,mode='min'
    ,verbose=1
    ,patience=3
)

adapted_lr = 2e-5
#Todo hier die decay-function schreiben
#Learning rate decay function used by Callback
def learningrate_decay(epoch,lr):
    print(global_train_loss)
    
    if epoch < 8:
        print(lr)
        return lr
    else:
        avg_train_loss = (global_train_loss[epoch-3]+global_train_loss[epoch-2]+global_train_loss[epoch-1])/3
        avg_val_loss = (global_val_loss[epoch-3]+global_val_loss[epoch-2]+global_val_loss[epoch-1])/3
        avg_residual_loss = avg_train_loss-avg_val_loss
        global adapted_lr
        print("Avg resid loss")
        print(avg_residual_loss)
        if avg_residual_loss < 0:
            print("Decay")
            adapted_lr = adapted_lr*(1-((-1)*avg_residual_loss))
        print("Adapted_lr")
        print(adapted_lr)
        return adapted_lr
        #global_loss ist der training loss, noch nicht der validation loss
        #print("global loss in epoche " + str(epoch))
        #print(global_loss)
    print("Outer Loss")
    print(global_train_loss)
    print(global_val_loss)

    

#Initialize LearningRateScheduler Callback
learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(learningrate_decay)

#Global lists with loss values for decay function
global_train_loss = []
global_val_loss = []

#Callback to track training and validation loss
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs=None):
        
        
        train_loss = logs.get('loss')
        global_train_loss.append(train_loss)

        val_loss = logs.get('val_loss')
        global_val_loss.append(val_loss)
        #print("global loss in epoche " + str(epoch))
        #print(global_loss)

#Initialising loss Callback
custom_lr_update_callback = CustomCallback()


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

    model_densenet201 = tf.keras.applications.DenseNet201(weights='imagenet',input_tensor = input_layer,include_top = False)

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
    
    input_layer = layers.Input(shape=(373,373,3))

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
    model_ResNet152V2.summary()
    model.summary()
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
    #keras.utils.plot_model(model_Xception,to_file='Xception.png',show_shapes=True,show_layer_names=False)
    #Layer untrainable machen
    for layer in model.layers[:-1]: #auf -1 ändern, wenn nur der finale classifier und keine Dense schicht
        layer.trainable=False

    model_compiler(model)

    print("Ab hier: Model Fitting")
    model_fitter(model,train_generator,validation_generator,weight)

    print("Ab hier: Model evaluation")
    model_evaluater(test_generator)

def InceptionResNetV2(train_generator,validation_generator,test_generator,weight):
    
    input_layer = layers.Input(shape=(373,373,3))

    model_InceptionResNetV2 = tf.keras.applications.InceptionResNetV2(weights='imagenet',input_tensor = input_layer,include_top = False)

    flatten = tf.keras.layers.Flatten()
    classifier = tf.keras.layers.Dense(1,activation='sigmoid')
    globalaverage = tf.keras.layers.GlobalAveragePooling2D()
    dense1 = tf.keras.layers.Dense(128,activation='relu')
    dropout = tf.keras.layers.Dropout(0.5)

    model = tf.keras.models.Sequential([
        model_InceptionResNetV2,
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

#Note: For EfficientNet, remove rescaling in generator and set targetsize to 300 300
def EfficientNetB4(train_generator,validation_generator,test_generator,weight):
    
    input_layer = layers.Input(shape=(380,380,3))

    model_EfficientNetB4 = tf.keras.applications.EfficientNetB4(weights='imagenet',input_tensor = input_layer,include_top = False)

    flatten = tf.keras.layers.Flatten()
    classifier = tf.keras.layers.Dense(1,activation='sigmoid')
    globalaverage = tf.keras.layers.GlobalAveragePooling2D()
    dense1 = tf.keras.layers.Dense(128,activation='relu')
    dropout = tf.keras.layers.Dropout(0.5)

    model = tf.keras.models.Sequential([
        model_EfficientNetB4,
        flatten,
        #globalaverage,
        #dense1,
        #dropout,
        classifier
    ])
    #Layer untrainable machen
    for layer in model.layers[:-1]: #auf -1 ändern, wenn nur der finale classifier und keine Dense schicht
        layer.trainable=False

    #keras.utils.plot_model(model_EfficientNetB4,to_file='EfficientNetB4.png',show_shapes=True,show_layer_names=True)
    model_compiler_old(model)

    print("Ab hier: Classifier Fitting")
    model_fitter(model,train_generator,validation_generator,weight,20)

  

    print("Ab hier: Model evaluation")
    model_evaluater(test_generator)

#Benchmarking has been done with model_compiler_old
def model_compiler_old(model):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=['accuracy','FalseNegatives','FalsePositives','Precision','Recall'])

#optimized learning rate for hypertuning EfficientNetB4
def model_compiler(model):
    initial_lr = 1e-4
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_lr,
        decay_steps=130,
        decay_rate=0.94,
        staircase=True
    )
    print("scheduler jetzt da")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=['accuracy','FalseNegatives','FalsePositives','Precision','Recall'])



def model_fitter(model,train_generator,validation_generator,weight,epochs):
    history = model.fit(train_generator,validation_data=validation_generator,epochs=epochs
                        ,class_weight = {0: 1, 1: weight}
                        ,callbacks =[model_callback,stopper,tensorboard_callback,learning_rate_scheduler,custom_lr_update_callback]
                        #,steps_per_epoch=500
                        )
#Todo den loss der class an die learningrate decay übergeben

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

    
    