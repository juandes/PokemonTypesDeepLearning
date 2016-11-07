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
import random
import pandas as pd



IMAGE_SIZE = 32


def load_images():
    """
    Returns a tuple made of an array of 18 (types) datasets of the shape 
    (len(type_image), IMAGE_SIZE, IMAGE_SIZE, 3), another array that has 
    the labels (Pokemon type), and an array made of the name of the Pokemon
    """
    labels = []
    pokemon_name = []
    image_index = 0
    # 714 because the Flying Pokemon were removed
    dataset = np.ndarray(shape=(714, IMAGE_SIZE, IMAGE_SIZE, 3),
                            dtype=np.float32)
    # Loop through all the types directories
    for type in os.listdir('./Images/data'):
        type_images = os.listdir('./Images/data/' + type + '/')
        # Loop through all the images of a type directory
        for image in type_images:
            image_file = os.path.join(os.getcwd(), 'Images/data', type, image)
            pokemon_name.append(image)
            # reading the images as they are; no normalization, no color editing
            image_data = (ndimage.imread(image_file, mode='RGB'))
            if image_data.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
                raise Exception('Unexpected image shape: %s %s' % (str(image_data.shape), image_file))
            dataset[image_index, :, :] = image_data
            image_index += 1
            labels.append(type)
        
    return (dataset, labels, pokemon_name)


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

    # Print the shape of the tensor
    print ("Convolution layer 1 shape: {}".format(h_conv1.get_shape()))
    h_pool1 = max_pool_2x2(h_conv1)
    print ("Convolution layer 1 shape (after pooling): {}".format(h_pool1.get_shape()))

  # second conv layer will compute 64 features for each 5x5 patch.
  with tf.variable_scope('conv_layer2'):
    h_conv2 = learn.ops.conv2d(h_pool1, n_filters=64, filter_shape=[5, 5],
                               bias=True, activation=tf.nn.relu)
    print ("Convolution layer 2 shape: {}".format(h_conv2.get_shape()))

    h_pool2 = max_pool_2x2(h_conv2)
    print ("Convolution layer 2 shape (after pooling): {}".format(h_pool2.get_shape()))

    # reshape tensor into a batch of vectors
    h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 64])
    print ("Final shape: {}", h_pool2_flat.get_shape())
  # densely connected layer with 1024 neurons.
  h_fc1 = learn.ops.dnn(
      h_pool2_flat, [1024], activation=tf.nn.relu, dropout=0.5)
  return learn.models.logistic_regression(h_fc1, y)


def main():
    images, labels, pokemon = load_images()
    pokemon_test = []
    print (labels)

    # Label encoder
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    print (le.classes_)
    transformed_labels = le.transform(labels)
    print (transformed_labels)

    msk = np.random.rand(714) < 0.8
    print (msk)
    true_indexes = []
    false_indexes = []
    training_labels = []
    test_labels = []
    for idx, val in enumerate(msk):
        if val == 1:
            true_indexes.append(idx)
            training_labels.append(transformed_labels[idx])
        else:
            false_indexes.append(idx)
            test_labels.append(transformed_labels[idx])
            pokemon_test.append(pokemon[idx])

    training_set = np.delete(images, false_indexes, 0)
    test_set = np.delete(images, true_indexes, 0)


    reshaped_dataset = training_set.reshape(len(training_labels), 3072)
    reshaped_testset = test_set.reshape(len(test_labels), 3072)

    # Training and predicting.
    classifier = learn.TensorFlowEstimator(
        model_fn=conv_model, n_classes=17, batch_size=100, steps=20000,
        learning_rate=0.001, verbose=2)
    classifier.fit(reshaped_dataset, training_labels, logdir=os.getcwd() + 'model_20000b_logs')
    classifier.save(os.getcwd() + '/model_20000b')
    score = metrics.accuracy_score(
        test_labels, classifier.predict(reshaped_testset))
    print('Accuracy: {0:f}'.format(score))

    prediction_labels = classifier.predict(reshaped_testset)
    target_names=['Bug' 'Dark' 'Dragon' 'Electric' 'Fairy' 'Fighting' 'Fire' 'Ghost' 'Grass'
        'Ground' 'Ice' 'Normal' 'Poison' 'Psychic' 'Rock' 'Steel' 'Water']

    print (metrics.classification_report(test_labels, prediction_labels))
    print (test_labels)
    print (prediction_labels)
    print (pokemon_test)


if __name__ == "__main__":
    main()
