#region imports and settings
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import tensorflow as tf
import xml.etree.ElementTree as ET

from Models import *

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
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

    #Navigiert path in Bilder Ordner
    os.chdir(os.path.join(os.getcwd(),'Dataset_FracAtlas','images','all'))
    

    #Plots erstellen, Laden  des Bildes
    fig,ax = plt.subplots()
    ax.imshow(mpimg.imread(name),cmap='gray')

    #cwd zurück auf Standard-Ordner setzen
    os.chdir(os.path.join(os.getcwd(),'..','..','..'))
    
    #Falls Bruch, wird boundingbox auf plot gelegt
    if general_info_df.loc[general_info_df['image_id'] == name, 'fractured'].values == 1:
        rectangle = boundingbox(name,fig,ax)
        ax.add_patch(rectangle)

    #zeigt PLot an
    plt.show()
    
#Todo 
def image_preprocessing(name):
    pass


def boundingbox(name,fig,ax):
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


#Todo nicht zu viele negative bilder rauslöschen, loss function anpassen
def csv_preprocessing(df):
    """
    Creating csv with Columns ['image_id','fractured'] for dataset input pipeline. Args[df: main dataframe]
    """

    #Datensatz eingegrenzt
    #df = df[df['leg'] == 1]
    df = df[df['hardware'] == 0]
    df = df[(df['fracture_count'] == 0) | (df['fracture_count'] == 1)]


    #inital main dataframe turned into dataset with columns 'image_id' and str('fractured')
    dataset = df[['image_id', 'fractured']].assign(fractured=df['fractured'].astype(str))

    dataset = dataset.sample(frac = 0.2)

    #!Gewicht, da non-fractured und fractured nicht gleich viele. Wird übergeben in model
    gewicht = int(round(((dataset['fractured'].value_counts()).get(0,1))/(dataset['fractured'].value_counts()).get(1,0),0))
    print(gewicht)
    return dataset, gewicht


#Todo k-fold cross validation einbauen. Mal gucken wo das rein muss
def create_generators(df,targetsize):
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
                                                        target_size=targetsize,
                                                        subset='training'
                                                        ,batch_size=32)

    validation_generator = datagen.flow_from_dataframe(dataframe=df,
                                                        directory=path,
                                                        x_col='image_id',
                                                        y_col='fractured',
                                                        class_mode='binary',
                                                        target_size=targetsize,
                                                        subset='validation'
                                                        ,batch_size=32)
    return train_generator, validation_generator



    

def main():
    pipeline_dataframe, gewicht = csv_preprocessing(general_info_df)

    
    targetsize = (373,373)
    train_generator,validation_generator = create_generators(pipeline_dataframe,targetsize)
    print(pipeline_dataframe)
    model_CNN(train_generator,validation_generator,gewicht)


    #showimage('IMG0000057.jpg')

if __name__ == "__main__":
    main()