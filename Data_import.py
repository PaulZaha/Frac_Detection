
#!in PASCAL VOC annotations sind bounding boxes für fractures drin!!!

#region imports and settings
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf

from Models import *

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import xml.etree.ElementTree as ET
#endregion


#region reading initial main dataframe
#Set working directory to the dataset folder
os.chdir(os.path.join(os.getcwd(),'Dataset_FracAtlas'))
#Einlesen der dataset.csv Datei
general_info_df = pd.read_csv('dataset.csv')
#cwd zurück zu ProgrammingDigitalProjects setzen
os.chdir(os.path.join(os.getcwd(),'..'))
#endregion

def showimage(name):
    """
    Shows images using matplotlib package. Args[name: 'image_id.jpg']
    """

    #Navigiert path in fractured, bzw. Non_fractured Ordner.
    os.chdir(os.path.join(os.getcwd(),'Dataset_FracAtlas','images','all'))
    rows = 5
    columns = 5

    #Laden und Anzeigen des Bildes
    plt.imshow(mpimg.imread(name),cmap='gray')
    plt.show()
    
    #cwd zurück auf Standard-Ordner setzen
    os.chdir(os.path.join(os.getcwd(),'..','..','..'))
    
def boundingbox(name,df):
    #Todo prüfen, ob Bruch vorhanden, noch nicht mit drin
    #!Prof. Büttner fragen, ob das überhaupt sinn macht, in jpgs reinzuzoomen
    path = os.path.join(os.getcwd(),'Dataset_FracAtlas','Annotations','PASCAL VOC')
    os.chdir(path)
    #print(path)
    tree = ET.parse(name[:-3]+'xml')
    root = tree.getroot()
    
    
    
#Todo Hardware & mixed pictures entfernen
#Todo Fracture Count = 0 oder =1, Rest rauslöschen
def csv_preprocessing(df):
    """
    Creating csv with Columns ['image_id','fractured'] for dataset input pipeline. Args[df: main dataframe]
    """

    #Datensatz eingegrenzt auf leg
    df = df[df['leg'] == 1]
    df = df.sample(frac = 1)
    print(df)

    #inital main dataframe turned into dataset with columns 'image_id' and str('fractured')
    dataset = df[['image_id', 'fractured']].assign(fractured=df['fractured'].astype(str))
    

    #Datensatz auf ähnliche Anzahl 0 und 1 aufteilen.
    #dataset = dataset.sample(frac = 1)
    boolean_index = (dataset['fractured'] == '0')
    to_delete = dataset[boolean_index].head(1749)
    dataset = dataset.drop(to_delete.index)
    dataset = dataset.sample(frac = 1)
    #print(dataset['fractured'].sum())
    

    #print(dataset['fractured'].sum())
    return dataset

def create_generators(df):
    """
    Creates a training image dataset and a validation image dataset. Args[df: preprocessed dataframe]
    """
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255, validation_split=0.2)
    path = os.path.join(os.getcwd(),'Dataset_FracAtlas','images','all')

    train_generator = datagen.flow_from_dataframe(dataframe=df,
                                                        directory=path,
                                                        x_col='image_id',
                                                        y_col='fractured',
                                                        class_mode='binary',
                                                        target_size=(224,224),
                                                        subset='training'
                                                        ,batch_size=16)

    validation_generator = datagen.flow_from_dataframe(dataframe=df,
                                                        directory=path,
                                                        x_col='image_id',
                                                        y_col='fractured',
                                                        class_mode='binary',
                                                        target_size=(224,224),
                                                        subset='validation'
                                                        ,batch_size=16)
    return train_generator, validation_generator



def image_preprocessing():
    
    pass

def main():
    pipeline_dataframe = csv_preprocessing(general_info_df)

    #boundingbox('IMG0002434.jpg',pipeline_dataframe)
    #train_generator,validation_generator = create_generators(pipeline_dataframe)
    print(pipeline_dataframe)
    #model_CNN(train_generator,validation_generator)

    #showimage('IMG0002434.jpg')

if __name__ == "__main__":
    main()