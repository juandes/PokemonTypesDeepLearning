import matplotlib.pyplot as plt

def first_image_per_type(dataset):
    for type in dataset:
        plt.imshow(type[0])
        plt.show()