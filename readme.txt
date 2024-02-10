Respository for paper "Improving deep learning x-ray fracture detection with a dynamic loss-linked learning-rate decay function"

Achieving an accuracy of 93.55% for multiple bone regions, including all radiological views with and without inserted hardware, setting a new benchmark in this field.


Quick repository manual:

Step 1: Image_preaugmentation.py
Execute script, comments are in script. Augments images in respective folders.

Step 2: Dataframe_preaugmentation.py
Must be executed after step 1, as exactly those images are thrown out of the dataframe that are removed from the directory in step 1 are removed from the directory.

Step 3: pipelines.py
Currently fitted to EfficientNetB4 (with targetsize=380x380 in main method). Can be adjusted to compare to other models.

Explanation main in pipelines.py:
randomized splitting in train and test dataset
Create the generators
with e.g. "EfficientNetB4(train_generator,validation_generator,test_generator,weight)" (line 184) model can be trained and evaluated.
Access to models.py

Method of the respective model is called in models.py
All callbacks, as well as compiling, fitting and evaluation are called in the model method.
