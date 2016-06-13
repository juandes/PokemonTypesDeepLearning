import os
from scipy import ndimage

image_size = 320

# Load images
image_folders = os.listdir('./Images/Resized')


for type in image_folders:
    for image in os.listdir('./Images/Resized/'+type+'/'):
        image

