from PIL import Image
import os
input_directory = 'C:/Users/paulz/Documents/ProgrammingDigitalProjects/Dataset_FracAtlas/images/all_test'
output_directory = 'C:/Users/paulz/Documents/ProgrammingDigitalProjects/Dataset_FracAtlas/images/resized'

target_size = (373,373)

for filename in os.listdir(input_directory):
    if filename.endswith(".jpg") or filename.endswith(".jpeg"):
        img_path = os.path.join(input_directory, filename)

        img = Image.open(img_path)
        img = img.resize(target_size)


        output_path = os.path.join(output_directory,filename)
        img.save(output_path)