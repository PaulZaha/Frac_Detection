import os
import pandas as pd

#PZ: Für euch geht das noch nicht, da das Dataset noch auf meinem Laptop liegt. 
#Wird in relative Pathing geändert, wenn wir uns für ein Dataset entschieden haben
os.chdir('C:/Users/paulz/OneDrive - Universität Bayreuth/#Uni/Master/Semester 1/Programmieren für digitale Projekte/Datasets/FracAtlas')

#Dataframe initialisieren
dataframe = pd.read_csv('dataset.csv')

#Dict mit Körperteil als Key und Anzahl der Brüche als Value
anzahl_by_location = {}
anzahl_by_type = {}

#Function um Pairs zu dict zu adden
def counter_bodyparts(column):
    anzahl_by_location[column] = dataframe[column].sum()

def counter_fracture(column):
    anzahl_by_type[column] = dataframe[column].sum()

#iterating over body parts in dataframe
for bodyparts in list(dataframe.columns[1:6]):
    counter_bodyparts(bodyparts)

#iterating over fracture type in dataframe
for type in list(dataframe.columns[10:13]):
    counter_fracture(type)

print(dataframe)
print(anzahl_by_location)
print(anzahl_by_type)