import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import nnfs


nnfs.init()


# labels = os.listdir('fashion_mnist_images/train')
# print(labels)
#
# files = os.listdir('fashion_mnist_images/train/0')
# print(files[:10])
# print(len(files))

# image_data = cv2.imread('fashion_mnist_images/train/7/0002.png',
#                         cv2.IMREAD_UNCHANGED)
# np.set_printoptions(linewidth=200)
# print(image_data)

# image_data = cv2.imread('fashion_mnist_images/train/4/0011.png',
#                         cv2.IMREAD_UNCHANGED)
#
# plt.imshow(image_data)
# plt.show()

# image_data = cv2.imread('fashion_mnist_images/train/4/0011.png',
#                         cv2.IMREAD_UNCHANGED)
#
# plt.imshow(image_data, cmap='gray')
# plt.show()

# Loads a MNIST dataset
def load_mnist_dataset(dataset, path):

    # Scan all the directories and create a list of labels
    labels = os.listdir(os.path.join(path, dataset))

    # Create lists for samples and labels
    X = []
    y = []


    # For each label folder
    for label in labels:
        # And for each image in given folder
        for file in os.listdir(os.path.join(path, dataset, label)):
            # Read the image
            image = cv2.imread(
                        os.path.join(path, dataset, label, file),
                        cv2.IMREAD_UNCHANGED)

            # And append it and a label to the lists
            X.append(image)
            y.append(label)

    # Convert the data to proper numpy arrays and return
    return np.array(X), np.array(y).astype('uint8')


# MNIST dataset (train + test)
def create_data_mnist(path):

    # Load both sets separately
    X, y = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)

    # And return all the data
    return X, y, X_test, y_test


# Create dataset
X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')

# Shuffle the training dataset
keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]

# Scale and reshape samples
X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) -
             127.5) / 127.5

# print(X.min(), X.max())
# print(X.shape)
# print(y[0:10])

plt.imshow((X[8].reshape(28, 28))) # Reshape as image is a vector already
plt.show()

# Check the class at the same index
# print(y[8])
