import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Set working directory to the dataset folder
dataset_path = os.path.join(os.getcwd(),'Dataset_FracAtlas')
os.chdir(dataset_path)

#Einlesen der dataset.csv Datei
general_info_df = pd.read_csv('dataset.csv')
#print(general_info_df)



def data_split(type):
    """
    Grenzt den general_info_Dataframe ein. Type kann test, train oder valid sein.
    """
    dataset_temp = pd.read_csv(os.path.join(dataset_path,'Utilities','Fracture Split',type + '.csv'))
    dataset = general_info_df[general_info_df['image_id'].isin(dataset_temp['image_id'])]
    return dataset


def main():
    print(data_split('train'))

if __name__ == "__main__":
    main()