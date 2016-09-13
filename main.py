from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from scipy import ndimage
from sklearn import metrics, preprocessing
from tensorflow.contrib import learn

import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


IMAGE_SIZE = 32


def load_images2():
    """
    Returns a tuple made of an array of 18 (types) datasets of the shape 
    (len(type_image), IMAGE_SIZE, IMAGE_SIZE, 3) and another array that has 
    the labels (Pokemon type)
    """
    labels = []
    image_index = 0
    dataset = np.ndarray(shape=(717, IMAGE_SIZE, IMAGE_SIZE, 3),
                            dtype=np.float32)
    for type in os.listdir('./Images/Resized'):
        type_images = os.listdir('./Images/Resized/' + type + '/')
        for image in type_images:
            image_file = os.path.join(os.getcwd(), 'Images/Resized', type, image)
            # reading the images as they are; no normalization, no color editing
            image_data = (ndimage.imread(image_file, mode='RGB'))
            if image_data.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
                raise Exception('Unexpected image shape: %s %s' % (str(image_data.shape), image_file))
            #plt.imshow(image_data)
            #plt.show()
            dataset[image_index, :, :] = image_data
            image_index += 1
            labels.append(type)
            #dataset_names.append(dataset.astype(int))
        
    return (dataset, labels)


def load_images(image_size):
    """
    Returns a tuple made of an array of 18 (types) datasets of the shape 
    (len(type_image), IMAGE_SIZE, IMAGE_SIZE, 3) and another array that has 
    the labels (Pokemon type)
    """
    dataset_names = []
    labels = []
    for type in os.listdir('./Images/Resized'):
        image_index = 0
        type_images = os.listdir('./Images/Resized/' + type + '/')
        dataset = np.ndarray(shape=(len(type_images), IMAGE_SIZE, IMAGE_SIZE, 3),
                            dtype=np.float32)
        labels.append(type)
        for image in type_images:
            image_file = os.path.join(os.getcwd(), 'Images/Resized', type, image)
            # reading the images as they are; no normalization, no color editing
            image_data = (ndimage.imread(image_file, mode='RGB'))
            if image_data.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
                raise Exception('Unexpected image shape: %s %s' % (str(image_data.shape), image_file))
            dataset[image_index, :, :] = image_data
            image_index += 1
            #dataset_names.append(dataset.astype(int))
        dataset_names.append(dataset.astype(int))
        
    return (dataset_names, labels)


def first_image_per_type(dataset):
    for pokemon in dataset:
        plt.imshow(pokemon)
        plt.show()

        ### Convolutional network

def max_pool_2x2(tensor_in):
  return tf.nn.max_pool(
      tensor_in, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def conv_model(X, y):
  # pylint: disable=invalid-name,missing-docstring
  # reshape X to 4d tensor with 2nd and 3rd dimensions being image width and
  # height final dimension being the number of color channels.
  X = tf.reshape(X, [-1, IMAGE_SIZE, IMAGE_SIZE, 3])
  # first conv layer will compute 32 features for each 5x5 patch
  with tf.variable_scope('conv_layer1'):
    h_conv1 = learn.ops.conv2d(X, n_filters=32, filter_shape=[5, 5],
                               bias=True, activation=tf.nn.relu)
    h_pool1 = max_pool_2x2(h_conv1)
  # second conv layer will compute 64 features for each 5x5 patch.
  with tf.variable_scope('conv_layer2'):
    h_conv2 = learn.ops.conv2d(h_pool1, n_filters=64, filter_shape=[5, 5],
                               bias=True, activation=tf.nn.relu)
    h_pool2 = max_pool_2x2(h_conv2)

    # reshape tensor into a batch of vectors
    h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 64])
  # densely connected layer with 1024 neurons.
  h_fc1 = learn.ops.dnn(
      h_pool2_flat, [1024], activation=tf.nn.relu, dropout=0.5)
  return learn.models.logistic_regression(h_fc1, y)


def main():
    images, labels = load_images2()
    print (labels)
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    print (le.classes_)
    transformed_labels = le.transform(labels)
    print (transformed_labels)

    #print type(images)
    #print images.shape
    #print labels
    #print images[0,0, :]
    #print images.shape()
    # The 717 is because there are 717 images. Trust me.
    #dataset = np.ndarray((717, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    #dataset = images.reshape((717, IMAGE_SIZE * IMAGE_SIZE))
    #first_image_per_type(images)
    reshaped_dataset = images.reshape(717, 3072)
    print (reshaped_dataset.shape)
    # Training and predicting.
    classifier = learn.TensorFlowEstimator(
        model_fn=conv_model, n_classes=18, batch_size=100, steps=2000,
        learning_rate=0.001)
    classifier.fit(reshaped_dataset, transformed_labels)
    #score = metrics.accuracy_score(
     #   mnist.test.labels, classifier.predict(mnist.test.images))
    #print('Accuracy: {0:f}'.format(score))
    #print reshaped_dataset[0, 50:70]
    #print reshaped_dataset.shape



if __name__ == "__main__":
    main()