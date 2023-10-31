import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

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

#Todo: vektoren durch 255 teilen, scaling anpassen
#Todo preprocessing in extra file schieben. Neuronales Netz in eigene File, dass es wiederverwendbar ist
def preprocessing():
    #Todo Überlegen, ob das nicht in create_dataset() mit rein kommt
    pass




#Todo noch für train-Datensatz angelegt. Muss noch aufgeteilt werden.
def create_dataset():
    """
    Erstellt einen TensorFlow-Datensatz. Einput ist ein dataframe (training, test oder valid datensatz)
    """

    #Todo rescale stimmt nicht, das sind nicht die RGB-values sondern die x und y achse
    images_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)
    path = os.path.join(os.getcwd(),'Dataset_FracAtlas','images')
    images, labels = next(images_generator.flow_from_directory(path,target_size=(28,28),color_mode='grayscale'))

    return images, labels





def main():
    training_dataset = data_split('train')
    #print(training_dataset)

    images, labels = create_dataset()

    print(images.shape)


    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    
    model.fit(images,labels, epochs=5)




    #plt.imshow(images[1,:,:])
    #plt.show()
    #print(labels[7])





    #showimage('IMG0000000.jpg')
    #print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    #print(create_dataset(training_dataset))

if __name__ == "__main__":
    main()