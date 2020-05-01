import struct
import numpy as np

def loadMnistLabels(filename,toRead=-1):
    with open(filename, "rb") as f:
        f.read(4) #magic number
        numberOfImages = struct.unpack('>i', f.read(4))[0]
        if toRead == -1:
            toRead = numberOfImages
        labels = np.zeros((toRead,10))
        for i in range(toRead):
            digit = struct.unpack('<b', f.read(1))[0]
            labels[i][digit] = 1.0
            i += 1
    return labels

def loadMnistImages(filename,toRead=-1):
    with open(filename, "rb") as f:
        f.read(4) #magic number
        numberOfImages = struct.unpack('>i', f.read(4))[0]
        rows = struct.unpack('>i', f.read(4))[0]
        cols = struct.unpack('>i', f.read(4))[0]
        if toRead == -1:
            toRead = numberOfImages;
        buf = f.read(rows * cols * toRead)
        images = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        images = images.reshape(toRead, rows*cols)
        images /= 255
    return images

def displayMnistDigit(digit):
    for i in range(len(digit)):
        if (i%28) == 0 : print("|")
        if digit[i] > 0.0 : print("O", end='')
        else: print(" ", end='')
