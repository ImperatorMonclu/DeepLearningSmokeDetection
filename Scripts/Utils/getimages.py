from cv2 import imread, imwrite
from numpy import where, asarray, zeros, array
from json import load

from os.path import join
from glob import glob

from os.path import exists
from os import mkdir


def getimages(relativePath, classNames):
    filenameImages = list(glob(join(relativePath, 'SegmentetedImages', 'labels', '*')))

    filenameJSONs = list(
        glob(join(relativePath, 'LabelsJSON', '*')))

    imagesDir = join(relativePath, 'Images')
    if not exists(imagesDir):
        mkdir(imagesDir)

    labelsDir = join(relativePath, 'Labels')
    if not exists(labelsDir):
        mkdir(labelsDir)
    for c in classNames:

    filenameJSONs = list(
        glob(join(relativePath, 'LabelsJSON', '*')))

    if not exists(images_train_dir):
        mkdir(images_train_dir)

    images_test_dir = join(relative_path, 'data', 'test1', 'images')
    if not exists(images_test_dir):
        mkdir(images_test_dir)

    filenames = []
    for f in filenames_train:
        filenames.append((f.split('\\')[-1]).split('.')[0])

    for f in filenames_train_data:
        if (f.split('\\')[-1]).split('.')[0] in filenames:
            imwrite(join(images_train_dir, f.split('\\')[-1]), imread(f))
        else:
            imwrite(join(images_train_dir2, f.split('\\')[-1]), imread(f))


    return filenames_test_data

getimages(join('C:\\', 'Users', 'mauro', 'Documents', 'Minsait', 'NoOneDrive', 'UNet'))
