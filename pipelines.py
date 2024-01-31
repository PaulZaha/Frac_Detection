#region imports and settings
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import tensorflow as tf
import xml.etree.ElementTree as ET

import sklearn
from sklearn.model_selection import KFold, train_test_split

from Models import *

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
#endregion
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#region reading initial main dataframe
#Set working directory to the dataset folder
os.chdir(os.path.join(os.getcwd(),'Dataset_FracAtlas'))
#Einlesen der dataset.csv Datei
general_info_df = pd.read_csv('dataset_preaugmented.csv')
#cwd zurück zu ProgrammingDigitalProjects setzen
os.chdir(os.path.join(os.getcwd(),'..'))
#endregion

def showimage(name):
    """
    Shows images using matplotlib package. Args[name: 'image_id.jpg']
    """

    #Navigiert path in Bilder Ordner
    os.chdir(os.path.join(os.getcwd(),'Dataset_FracAtlas','images','all'))
    
    #Plots erstellen, Laden  des Bildes
    fig,ax = plt.subplots()
    ax.imshow(mpimg.imread(name),cmap='gray')
    plt.axis('off')
    #cwd zurück auf Standard-Ordner setzen
    os.chdir(os.path.join(os.getcwd(),'..','..','..'))
    
    #Falls Bruch, wird boundingbox auf plot gelegt
    if general_info_df.loc[general_info_df['image_id'] == name, 'fractured'].values == 1:
        rectangle = boundingbox(name,fig,ax)
        ax.add_patch(rectangle)

    #zeigt PLot an
    plt.show()
    
def boundingbox(name,fig,ax):
    """
    Shows boundingbox in showimage() if the picture is a fracture.#
    DANGER: Does not work for augmented fratures.
    """
    #Pathing in xml Ordner
    path = os.path.join(os.getcwd(),'Dataset_FracAtlas','Annotations','PASCAL VOC')
    os.chdir(path)

    #Tree initialisieren
    tree = ET.parse(name[:-3]+'xml')
    root = tree.getroot()

    #Pathing zurück auf Standard-Ordner
    os.chdir(os.path.join(os.getcwd(),'..','..','..'))
    
    #Werte aus bndbox element aus XML ziehen
    values = []
    for x in root[5][4]:
        values.append(int(x.text))
    
    #reassign value 2 und 3, da wir width & height brauchen statt 4 koordinaten für patches.rectangle
    values[2] = values[2]-values[0]
    values[3] = values[3]-values[1]

    #Rechteck wird festgelegt
    bounding_box = patches.Rectangle([values[0],values[1]],width=values[2],height=values[3],linewidth=1,edgecolor='r',facecolor='none')
    
    return bounding_box


def data_split(dataframe):
    """
    Returns train_dataset and test_dataset  dataframes as well as a weight factor for slightly unbalanced dataset.
    """

    #turn 'fractured' column to type(str)
    dataframe = dataframe[['image_id', 'fractured']].assign(fractured=dataframe['fractured'].astype(str))
    #dataframe = dataframe.sample(frac=0.1)
    #Datensatz aufgeteilt in 10% Testdaten und 90% Trainingsdaten
    train_dataset, test_dataset = train_test_split(dataframe, train_size = 0.9, shuffle = True)

    #!Gewicht kann am Ende rausgelöscht werden
    #Gewicht, da non-fractured und fractured nicht gleich viele. Wird übergeben in model
    gewicht = round(((dataframe['fractured'].value_counts()).get(0,1))/(dataframe['fractured'].value_counts()).get(1,0),3)

    return train_dataset, test_dataset,gewicht


def create_generators(train_df,test_df,targetsize):
    """
    Creates train_, validation_, and test_generator to feed Model with batches of data
    """
    #images pathing
    path = os.path.join(os.getcwd(),'Dataset_FracAtlas','images','edge_detection')

    #Create DataGenerator for training and validation data with augmentation
    #Note: For EfficientNet remove rescaling
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(#rescale = 1./255,
                                                              rotation_range=10 
                                                              ,width_shift_range=0.05
                                                              ,height_shift_range=0.05
                                                              ,validation_split=0.2)
    
    #Create DataGenerator for testing without augmentation
    #Note: For EfficientNet remove rescaling
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(#rescale=1./255
                                                                   )


    train_generator = datagen.flow_from_dataframe(dataframe=train_df,
                                                        directory=path,
                                                        x_col='image_id',
                                                        y_col='fractured',
                                                        class_mode='binary',
                                                        color_mode = 'rgb',
                                                        shuffle=True,
                                                        target_size=targetsize,
                                                        subset='training'
                                                        ,batch_size=32)
    

    validation_generator = datagen.flow_from_dataframe(dataframe=train_df,
                                                        directory=path,
                                                        x_col='image_id',
                                                        y_col='fractured',
                                                        class_mode='binary',
                                                        color_mode = 'rgb',
                                                        shuffle=True,
                                                        target_size=targetsize,
                                                        subset='validation'
                                                        ,batch_size=32)

    test_generator = test_datagen.flow_from_dataframe(dataframe=test_df,
                                                      directory=path,
                                                        x_col='image_id',
                                                        y_col='fractured',
                                                        class_mode='binary',
                                                        color_mode = 'rgb',
                                                        shuffle=True,
                                                        target_size=targetsize,
                                                        batch_size=1)

    return train_generator, validation_generator, test_generator


def main():
    #Create train and test split
    train_dataset, test_dataset, gewicht = data_split(general_info_df)

    showimage('IMG0000847.jpg')
    #Set targetsize
    targetsize = (380,380)
    
    #Create generators from train/test-split with chosen targetsize
    #train_generator,validation_generator, test_generator = create_generators(train_dataset,test_dataset,targetsize)

    #Feed generators to model from models.py
    

    #print("Densenet201")
    #densenet201(train_generator,validation_generator,test_generator,gewicht)
    #print("InceptionV3")
    #InceptionV3(train_generator,validation_generator,test_generator,gewicht)
    #print("ResNet152V2")
    #ResNet152V2(train_generator,validation_generator,test_generator,gewicht)
    #print("Xception")
    #Xception(train_generator,validation_generator,test_generator,gewicht)
    #print("InceptionResNetV2")
    #InceptionResNetV2(train_generator,validation_generator,test_generator,gewicht)
    #print("EfficientNetB4 380")
    #EfficientNetB4(train_generator,validation_generator,test_generator,gewicht)
    #Efficientnet funktioniert nicht, wieso auch immer. Mal die Errors anschauen

if __name__ == "__main__":
    main()