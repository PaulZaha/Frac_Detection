import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf




#Set working directory to the dataset folder
os.chdir(os.path.join(os.getcwd(),'Dataset_FracAtlas'))
#Einlesen der dataset.csv Datei
general_info_df = pd.read_csv('dataset.csv')

#cwd zurück zu ProgrammingDigitalProjects setzen
os.chdir(os.path.join(os.getcwd(),'..'))


def data_split(type):
    """
    Grenzt den general_info_Dataframe ein. Type kann test, train oder valid sein.
    """
    dataset_temp = pd.read_csv(os.path.join(os.getcwd(),'Dataset_FracAtlas','Utilities','Fracture Split',type + '.csv'))
    dataset = general_info_df[general_info_df['image_id'].isin(dataset_temp['image_id'])]
    return dataset




#Todo: vektoren durch 255 teilen, scaling anpassen
#Todo preprocessing in extra file schieben. Neuronales Netz in eigene File, dass es wiederverwendbar ist


def preprocessing():
    pass


def showimage(name):
    """
    Zeigt Bilder an. Eingabe Fractured / Non_fractured und Bildname.
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

#Todo https://stackoverflow.com/questions/37340129/tensorflow-training-on-my-own-image
#Todo dataset erstellen
def create_dataset():
    pass

def main():
    training_dataset = data_split('train')
    print(training_dataset)

    showimage('IMG0000000.jpg')

if __name__ == "__main__":
    main()