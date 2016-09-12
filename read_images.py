import os
import collections
import numpy as np

from scipy import ndimage



def load_images(image_size):
    """
    Returns an array of 18 (types) datasets of the shape (len(type_image), image_size, image_size, 3)
    """
    dataset_names = []
    for type in os.listdir('./Images/Resized'):
        image_index = 0
        type_images = os.listdir('./Images/Resized/' + type + '/')
        dataset = np.ndarray(shape=(len(type_images), image_size, image_size, 3),
                            dtype=np.float32)
        for image in type_images:
            image_file = os.path.join(os.getcwd(), 'Images/Resized', type, image)
            # reading the images as they are; no normalization, no color editing
            image_data = (ndimage.imread(image_file, mode='RGB'))
            if image_data.shape != (image_size, image_size, 3):
                raise Exception('Unexpected image shape: %s %s' % (str(image_data.shape), image_file))
            dataset[image_index, :, :] = image_data
            image_index += 1
        dataset_names.append(dataset.astype(int))
    
    return dataset_names

