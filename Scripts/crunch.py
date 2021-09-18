#Crunch image resolutions
#This script was designed for Machine Learning with Image datasets. Specifically facial recognition, using CNN'S algorithm uses a lot of CUDA memory on high quality images.
#Therefore I needed to compress the images when encoding a large batch.
#SCRIPT NEEDS TO BE IN THE SAME FOLDER AS THE DATASET. IF IT ISN'T THE DIRECTORIES NEED TO BE INCLUDED IN THE PATH
'''
This script/program is used for compressing datasets. 
The only requirement to run the script is a folder with your dataset, the script assumes sub directories are present.
Running the script will format all images in the resolution 540x540 and output them into their corresponding folders.
Example = dataset. dataset has subdirectories(1,2,3,4)
each subdirectory has an X amount of images
the compressed folder will contain (1,2,3,4)'s images in the same folder name but the images will all be 540x540 with IMAGENAMEcompressed.jpg
To remove the compressed value just modify the replace statement. Modify 'compressed' to '' 
'''
'''
RESULTS: Resolved a memory error with DLIB, speeded up recognition and made encoding a lot faster.
Testing recognition: Both datasets compressed and non compressed recognized the same amount of faces.
Both Using CNN
Identifying 32 images with 152 images to match = 15.92s COMPRESSED
Identifying 32 images with 152 images to match = 16.61s UNCOMPRESSED
'''
# https://pubs.rsna.org/doi/full/10.1148/ryai.2019190015
from pathlib import Path #> 3.4
import os
import sys
from PIL import Image
print("Compressing images")
SLASH = "/"
datasetPath = 'dataset/' #CHANGE ME
comp = 'compressed'
baseDir = Path(comp)
outputPath = comp + SLASH
directoryArray = os.listdir(datasetPath)
lengthArray = len(directoryArray)
if (baseDir.exists() == True):
    print("Directory already created.")
if (baseDir.exists() == False):
    os.mkdir('compressed')
for x in range(lengthArray):
    currentimageSub = os.listdir(datasetPath + directoryArray[x]) #Current image in the sub directory
    makeModifiedDir = Path(outputPath + directoryArray[x])
    if (makeModifiedDir.exists() == True):
        print("Directory already created.")
    if (makeModifiedDir.exists() == False):
        os.mkdir(makeModifiedDir)
    #print(makeModifiedDir)
    #print(type(currentimageSub))
    for currentImage in currentimageSub:
        #print(type(currentImage))
        partialPath = datasetPath + directoryArray[x] + SLASH
        completePath =  partialPath + currentImage
        #os.mkdir(makeModifiedDir)
        modifiedVal = outputPath + directoryArray[x] + SLASH + currentImage.replace('.jpg','compressed') + ".jpg"
        #print(type(changeVal))
        #print("ORIGINAL IMAGE: " + completePath)
        #print("COMPRESSED: " + modifiedVal)
        resizeImage = Image.open(completePath)
        RESIZE = (540, 540)
        resizeImage.thumbnail(RESIZE)
        resizeImage.save(modifiedVal)
    #print(listSub)
print("Finished compressing")