from PIL import Image, ImageFilter, ImageFile
import os
import shutil
import random

ImageFile.LOAD_TRUNCATED_IMAGES = True

from PIL.ImageFilter import (
   EDGE_ENHANCE_MORE
)

frac = 'C:/Users/paulz/Documents/ProgrammingDigitalProjects_v2/Dataset_FracAtlas/images/Fractured'

os.mkdir('C:/Users/paulz/Documents/ProgrammingDigitalProjects_v2/Dataset_FracAtlas/images/frac_augmented')
frac_augmented = 'C:/Users/paulz/Documents/ProgrammingDigitalProjects_v2/Dataset_FracAtlas/images/frac_augmented'


# #copy original fractures into frac_augmented folder
fracs = os.listdir(frac)
# #alle fracs 0.1 reinzoomen
# #alle bilder horizontal spiegeln
for filename in os.listdir(frac):
    img_path = os.path.join(frac, filename)
    filename = filename[:-4] + '_flipped.jpg'
    img = Image.open(img_path)

    flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)



    output_path = os.path.join(frac_augmented,filename)
    flipped_img.save(output_path)