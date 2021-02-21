import PIL
import cv2
import os

from PIL import Image

dirname="C:\\Users\\Adarsh\\PycharmProjects\\mask detector\\data"
dirs=os.listdir(dirname)


def resize():
    for subdirs,dirs,files in os.walk(dirname):
        for file in files:
            filepath=subdirs+os.sep+file

            if filepath.endswith(".jpg") :
                im = Image.open(filepath)
                #print(filepath)
                f=os.path.basename(filepath)
                imresize = im.resize((224, 224), Image.ANTIALIAS)
                if os.path.basename(subdirs)=='without_mask':
                    imresize.save("C:\\Users\\Adarsh\\PycharmProjects\\mask detector\\Without_mask\\"+f,"JPEG",quality=90)
                else:
                    imresize.save("C:\\Users\\Adarsh\\PycharmProjects\\mask detector\\With_mask\\"+f,"JPEG",quality=90)

resize()


