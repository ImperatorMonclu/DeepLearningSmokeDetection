from cv2 import imread, imwrite
from numpy import where, asarray, zeros, array
from json import load

from os.path import join
from glob import glob

from os.path import exists
from os import mkdir


def generate_labels(relative_path, labels):
    filenames_train = list(
        glob(join(relative_path, 'data', 'images', '*')))

    labels_train_dir = join(relative_path, 'data', 'labels')
    if not exists(labels_train_dir):
        mkdir(labels_train_dir)

    def write_images(list_filename, dir):
        for f in list_filename:
            dataimg = imread(f)
            img = zeros((dataimg.shape[0], dataimg.shape[1]), dtype='uint8')
            imwrite(join(dir, (f.split('\\')[-1]).split('.')[-2]+'.png'), img)
            
    write_images(filenames_train, labels_train_dir)


generate_labels(join('C:\\', 'Users', 'mauro', 'Documents',
                     'Minsait', 'NoOneDrive', 'aaa'), ['smoke'])
