from PIL import Image, ImageFilter
import os
input_directory = 'C:/Users/paulz/Documents/ProgrammingDigitalProjects_v2/Dataset_FracAtlas/images/edge_enhance_more'
output_directory = 'C:/Users/paulz/Documents/ProgrammingDigitalProjects_v2/Dataset_FracAtlas/images/doubled'

target_size = (373,373)

from PIL.ImageFilter import (
   BLUR, CONTOUR, DETAIL, EDGE_ENHANCE, EDGE_ENHANCE_MORE,
   EMBOSS, FIND_EDGES, SMOOTH, SMOOTH_MORE, SHARPEN
)


for filename in os.listdir(input_directory):
    if filename.endswith(".jpg") or filename.endswith(".jpeg"):
        img_path = os.path.join(input_directory, filename)
        #filename = filename[:-4] + '_zoomed.jpg'
        img = Image.open(img_path)

        breite, höhe = img.size
        neue_breite = int(breite*0.9)
        neue_höhe = int(höhe*0.9)
        links = (breite - neue_breite) // 2
        oben = (höhe - neue_höhe) // 2
        rechts = links + neue_breite
        unten = oben + neue_höhe

        img = img.filter(CONTOUR)


        output_path = os.path.join(output_directory,filename)
        img.save(output_path)