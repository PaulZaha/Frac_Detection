from PIL import Image
import os
input_directory = 'C:/Users/paulz/Documents/ProgrammingDigitalProjects_v2/Dataset_FracAtlas/images/full_augmented'
output_directory = 'C:/Users/paulz/Documents/ProgrammingDigitalProjects_v2/Dataset_FracAtlas/images/full_augmented_v2'

target_size = (373,373)

for filename in os.listdir(input_directory):
    if filename.endswith(".jpg") or filename.endswith(".jpeg"):
        img_path = os.path.join(input_directory, filename)
        filename = filename[:-4] + '_zoomed.jpg'
        img = Image.open(img_path)

        breite, höhe = img.size
        neue_breite = int(breite*0.9)
        neue_höhe = int(höhe*0.9)
        links = (breite - neue_breite) // 2
        oben = (höhe - neue_höhe) // 2
        rechts = links + neue_breite
        unten = oben + neue_höhe

        img = img.crop((links,oben,rechts,unten))


        output_path = os.path.join(output_directory,filename)
        img.save(output_path)