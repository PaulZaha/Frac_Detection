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
    os.chdir(os.path.join(os.getcwd(),'Dataset_FracAtlas','images','full_dataset'))
    rows = 5
    columns = 5

    #Laden und Anzeigen des Bildes
    plt.imshow(mpimg.imread(name),cmap='gray')
    plt.show()
    
    #cwd zurück auf Standard-Ordner setzen
    os.chdir(os.path.join(os.getcwd(),'..','..','..'))


def csv_preprocessing(df):
    """
    Creating csv with Columns ['image_id','fractured'] for dataset input pipeline. Args[df: main dataframe]
    """
    #inital main dataframe turned into dataset with columns 'image_id' and str('fractured')
    dataset = df[['image_id', 'fractured']].assign(fractured=df['fractured'].astype(str))
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
                                                        target_size=(64,64),
                                                        subset='training')

    validation_generator = datagen.flow_from_dataframe(dataframe=df,
                                                        directory=path,
                                                        x_col='image_id',
                                                        y_col='fractured',
                                                        class_mode='binary',
                                                        target_size=(64,64),
                                                        subset='validation')
    return train_generator, validation_generator



#Todo preprocessing in extra file schieben. Neuronales Netz in eigene File, dass es wiederverwendbar ist
def image_preprocessing():
    #Todo Überlegen, ob das nicht in create_dataset() mit rein kommt
    pass

def main():

    pipeline_dataframe = csv_preprocessing(general_info_df)
    train_generator,validation_generator = create_generators(pipeline_dataframe)

    
    model_CNN(train_generator,validation_generator)

    

if __name__ == "__main__":
    main()