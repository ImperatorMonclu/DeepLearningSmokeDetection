from os.path import join
from numpy.random import shuffle
from numpy import array
from glob import glob


# Carga las imágenes de prueba
def load(relativePath, n=256):
    filenameImagesTest = list(glob(join(relativePath, 'Test', '*')))
    shuffle(filenameImagesTest)
    if n >= len(filenameImagesTest):
        n = len(filenameImagesTest)
    return array(filenameImagesTest[:n]).reshape((n, 1))
