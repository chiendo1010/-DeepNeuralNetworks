import numpy as np
import matplotlib.pyplot as plt
import h5py
import struct





def load_data():
    train_set_x_orig = read_idx("datasets/train-images.idx3-ubyte") # your train set features
    train_set_y_orig = read_idx("datasets/train-labels.idx1-ubyte") # your train set labels


    test_set_x_orig = read_idx("datasets/t10k-images.idx3-ubyte") # your test set features
    test_set_y_orig = read_idx("datasets/t10k-labels.idx1-ubyte") # your test set label

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig


def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)
