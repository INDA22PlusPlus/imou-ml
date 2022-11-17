import numpy as np


# Parses the MNIST dataset format
class MNISTParser:
    # The file data
    _raw_train_labels = None
    _raw_train_images = None
    
    _raw_test_labels  = None
    _raw_test_images  = None

    def __init__(self, filename_train_labels, filename_train_images,
                filename_test_labels, filename_test_images):

        # Parse and check the train labels dataset
        with open(filename_train_labels, 'rb') as f:
            self._raw_train_labels = f.read()
            # Test the magic number
            if self._raw_train_labels[0:4] != b'\x00\x00\x08\x01':
                raise ValueError(f"{filename_train_labels} is corrupted")

        # Parse and check the train images dataset
        with open(filename_train_images, 'rb') as f:
            self._raw_train_images = f.read()
            # Test the magic number
            if self._raw_train_images[0:4] != b'\x00\x00\x08\x03':
                raise ValueError(f"{filename_train_images} is corrupted")

        # Parse and check the test labels dataset
        with open(filename_test_labels, 'rb') as f:
            self._raw_test_labels = f.read()
            # Test the magic number
            if self._raw_test_labels[0:4] != b'\x00\x00\x08\x01':
                raise ValueError(f"{filename_test_labels} is corrupted")

        # Parse and check the test images dataset
        with open(filename_test_images, 'rb') as f:
            self._raw_test_images = f.read()
            # Test the magic number
            if self._raw_test_images[0:4] != b'\x00\x00\x08\x03':
                raise ValueError(f"{filename_test_images} is corrupted")



    def get_train_label(self, index: int) -> int:
        return self._raw_train_labels[8+index]

    def get_test_label(self, index: int) -> int:
        return self._raw_test_labels[8+index]


    # Returns flattened and normalized numpy array with 784 pixel
    # values of the image
    def get_train_image(self, index: int) -> np.array:
        image = self._raw_train_images[(16+784*index):(16+784*(index+1))]
        return np.frombuffer(image, np.uint8, 784).astype(float)/255

    def get_test_image(self, index: int) -> np.array:
        image = self._raw_test_images[(16+784*index):(16+784*(index+1))]
        return np.frombuffer(image, np.uint8, 784).astype(float)/255