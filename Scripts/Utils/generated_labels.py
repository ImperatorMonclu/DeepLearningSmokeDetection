from cv2 import fillPoly, imwrite
from numpy import where, asarray, zeros, array
from json import load

from os.path import join
from glob import glob

from os.path import exists
from os import mkdir


def generate_labels(relative_path, labels):
    filenames_train = list(
        glob(join(relative_path, 'data', 'train', 'labels_json', '*')))

    filenames_test = list(
        glob(join(relative_path, 'data', 'test', 'labels_json', '*')))

    labels_train_dir = join(relative_path, 'data', 'train', 'labels')
    if not exists(labels_train_dir):
        mkdir(labels_train_dir)

    labels_test_dir = join(relative_path, 'data', 'test', 'labels')
    if not exists(labels_test_dir):
        mkdir(labels_test_dir)

def writeImages(listFilename, relativeDir, labels):
    for f in listFilename:
        with open(f) as jsonFile:
            data = load(jsonFile)

            shapesLabel = []
            for l in labels:
                shapesLabel.append(l)
                shapesLabel.append([])
            shapesLabel = array()

            for s in data['shapes']:
                shapesLabel[where(shapesLabel == s['label'])[
                    0][0]+1].append(asarray(s['points'], dtype=int))
            for i in range(len(shapes_bylabel)):
                img = zeros(
                    (data['imageHeight'], data['imageWidth']), dtype='uint8')
                if i % 2 == 0:
                    label_dir = join(relativeDir, shapesLabel[i])
                    if not exists(label_dir):
                        mkdir(label_dir)
                else:
                    imwrite(join(relativeDir, (f.split('\\')[-1]).split('.')[-2]+'.png'), fillPoly(
                        img, shapesLabel[i], color=255, lineType=1))


generate_labels(join('C:\\', 'Users', 'mgmonclu', 'Downloads',
                     'FotosFichasyPC'), ['0'])
