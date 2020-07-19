import struct
import numpy as np

def load_mnist_labels(filename, to_read=-1):
    """ Pass filename to mnist labels and pass 'to_read' - number of data to read.
        If 'to_read' is omitted or larger than number of data in mnist file,
        all data that mnist file contains will be read and returned.
        Remember, same 'to_read' value must be passed to 'load_mnist_images'. """
    with open(filename, "rb") as f:
        f.read(4) #magic number
        number_of_images = struct.unpack('>i', f.read(4))[0]
        if to_read == -1 or to_read > number_of_images:
            to_read = number_of_images
        labels = np.zeros((to_read, 10))
        for i in range(to_read):
            digit = struct.unpack('<b', f.read(1))[0]
            labels[i][digit] = 1.0
            i += 1
    return labels

def load_mnist_images(filename,to_read=-1):
    """ Pass filename to mnist images and pass 'to_read' - number of data to read.
        If 'to_read' is omitted or larger than number of data in mnist file,
        all data that mnist file contains will be read and returned.
        Remember, same 'to_read' value must be passed to 'load_mnist_labels'. """
    with open(filename, "rb") as f:
        f.read(4) #magic number
        number_of_images = struct.unpack('>i', f.read(4))[0]
        rows = struct.unpack('>i', f.read(4))[0]
        cols = struct.unpack('>i', f.read(4))[0]
        if to_read == -1:
            to_read = number_of_images;
        buf = f.read(rows * cols * to_read)
        images = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        images = images.reshape(to_read, rows*cols)
        images /= 255
    return images
