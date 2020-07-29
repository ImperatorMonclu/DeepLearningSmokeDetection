from os.path import join, exists
from os import sep, mkdir
from glob import glob
from numpy import hsplit, asarray, array, floor, arange, zeros, uint8, around, newaxis, concatenate, float32
from numpy.random import shuffle

from imgaug.augmenters import Sequential
from imgaug.augmenters.size import CenterPadToAspectRatio, Resize
from cv2 import INTER_AREA, imread, cvtColor, COLOR_BGR2RGB, IMREAD_UNCHANGED, imwrite, threshold, THRESH_BINARY, merge, bitwise_not, bitwise_and, bitwise_or, addWeighted, COLOR_RGB2BGR

from tensorflow.python.keras.utils.data_utils import Sequence
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from itertools import product

from datetime import datetime


# Crear el conjunto de datos con la dirección de las imágenes
def load(relativePath, labels, n=None, validationPercentage=None):

    filenameImagesTrain = list(
        glob(join(relativePath, 'Train', 'Images', '*')))
    pathListLabelsTrain = []
    for l in labels:
        pathListLabelsTrain.append(join(relativePath, 'Train', 'Labels', l))

    filenameImagesValidation = list(
        glob(join(relativePath, 'Validation', 'Images', '*')))
    pathListLabelsValidation = []
    for l in labels:
        pathListLabelsValidation.append(
            join(relativePath, 'Validation', 'Labels', l))

    if n is None:

        dataTrain = []
        for i in range(len(filenameImagesTrain)):
            data = []
            data.append(filenameImagesTrain[i])
            for l in pathListLabelsTrain:
                data.append(list(glob(join(l, ''.join(
                    (filenameImagesTrain[i].split(sep)[-1]).split('.')[:-1])+'.*')))[0])
            dataTrain.append(data)

        dataValidation = []
        if validationPercentage is None:
            for i in range(len(filenameImagesValidation)):
                data = []
                data.append(filenameImagesValidation[i])
                for l in pathListLabelsValidation:
                    data.append(list(glob(join(l, ''.join(
                        (filenameImagesValidation[i].split(sep)[-1]).split('.')[:-1])+'.*')))[0])
                dataValidation.append(data)
        else:
            for i in range(int(len(filenameImagesTrain)*validationPercentage)):
                data = []
                data.append(filenameImagesValidation[i])
                for l in pathListLabelsValidation:
                    data.append(list(glob(join(l, ''.join(
                        (filenameImagesValidation[i].split(sep)[-1]).split('.')[:-1])+'.*')))[0])
                dataValidation.append(data)
    else:
        if validationPercentage is None:
            validationPercentage = 0.5

        nTrain = int(n*(1.0-validationPercentage)/len(filenameImagesTrain))
        nValidation = int(n*validationPercentage/len(filenameImagesValidation))

        dataTrain = []
        for i in range(nTrain):
            data = []
            data.append(filenameImagesTrain[i])
            for l in pathListLabelsTrain:
                data.append(list(glob(join(l, ''.join(
                    (filenameImagesTrain[i].split(sep)[-1]).split('.')[:-1])+'.*')))[0])
            dataTrain.append(data)

        dataValidation = []
        for i in range(nValidation):
            data = []
            data.append(filenameImagesValidation[i])
            for l in pathListLabelsValidation:
                data.append(list(glob(join(l, ''.join(
                    (filenameImagesValidation[i].split(sep)[-1]).split('.')[:-1])+'.*')))[0])
            dataValidation.append(data)

    shuffle(dataTrain)
    shuffle(dataValidation)

    def splitLabelsImages(data, labels):
        dataSplit = hsplit(asarray(data), array(range(1, len(labels)+1)))
        images = dataSplit[0]
        labels = dataSplit[1:]
        return images, labels

    trainImages, trainLabels = splitLabelsImages(dataTrain, labels)

    validationImages, validationLabels = splitLabelsImages(
        dataValidation, labels)

    return trainImages, trainLabels, validationImages, validationLabels


# Filtro para redimensionar imágenes
def resize(imageDimensions):
    return Sequential(
        [
            CenterPadToAspectRatio(float(imageDimensions[0])/float(imageDimensions[1]),
                                   pad_mode='constant',
                                   pad_cval=0),
            Resize(imageDimensions,
                   interpolation=INTER_AREA)
        ])


# Da formato correcto a la máscara
def prepareLabels(indexes, imagesLabels):
    imagesLabelsList = []
    # Cambia el formato (tamaño del batch, número de clases, dimensión X, dimensión Y, 1)
    # a (tamaño del Batch, dimensión X, dimensión Y, número de clases)
    for k in range(len(indexes)):
        img = None
        for i in range(len(imagesLabels)):
            imgLabel = imagesLabels[i][k]
            if img is None:
                img = zeros(
                    (imgLabel.shape[0], imgLabel.shape[1], len(imagesLabels)), uint8)
            for x in range(len(imgLabel)):
                for y in range(len(imgLabel[x])):
                    img[x][y][i] = imgLabel[x][y]
        # Transforma los valores de 0-255 a 0.0-1.0
        imagesLabelsList.append(img/255.0)
    # Devuelve las máscaras convertidas a lista de floats
    return array(imagesLabelsList).astype('float32')


# Genera conjunto de datos para el entrenamiento
class DataGenerator(Sequence):

    def __init__(self,
                 imagePaths,
                 labelPaths,
                 batchSize=16,
                 resize=None,
                 augmenters=None,
                 shuffle=False):
        self.imagePaths = imagePaths
        self.labelPaths = labelPaths
        self.batchSize = batchSize
        self.resize = resize
        self.augmenters = augmenters
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(floor(len(self.imagePaths) / self.batchSize))

    def on_epoch_end(self):
        self.indexes = arange(len(self.imagePaths))
        if self.shuffle:
            shuffle(self.indexes)

    def __getitem__(self,
                    index):
        indexes = self.indexes[index *
                               self.batchSize: (index + 1) * self.batchSize]
        # Lee como imágenes las direcciones de ficheros que contiene el conjunto de datos
        images = [cvtColor(imread(str(self.imagePaths[k][0])),
                           COLOR_BGR2RGB) for k in indexes]
        imagesLabels = [[imread(str(l[k][0]), IMREAD_UNCHANGED)
                         for k in indexes] for l in self.labelPaths]
        if self.resize != None:
            # Realiza la redimensión correspondiente tanto a la imagen como a las máscaras
            images = self.resize.augment_images(images)
            imagesLabels = [[lsi.get_arr() for lsi in ls] for ls in [self.resize.augment_segmentation_maps(
                [SegmentationMapsOnImage(l[k], shape=l[k].shape) for k in range(len(indexes))]) for l in imagesLabels]]
        if self.augmenters != None:
            # Determina el data augmentation para que sea el mismo para cada imagen y sus correspondientes máscaras
            self.augmenters = self.augmenters.to_deterministic()
            images = self.augmenters.augment_images(images)
            imagesLabels = [[lsi.get_arr() for lsi in ls] for ls in [self.augmenters.augment_segmentation_maps(
                [SegmentationMapsOnImage(l[k], shape=l[k].shape) for k in range(len(indexes))]) for l in imagesLabels]]
        images = array([img/255.0 for img in images]).astype('float32')
        imagesLabels = prepareLabels(indexes, imagesLabels)
        return images, imagesLabels


# Genera una imagen con un grupo de imágenes
def imageGrid(images, labels, classNames=['0'], perRow=2, imageDimensions=(64, 64), nImages=16):
    nClasses = len(classNames)+1
    resizeImage = resize(imageDimensions)
    columns = perRow*nClasses
    rows = int((nImages*nClasses)/columns)
    figure, axarr = plt.subplots(rows, columns, figsize=(
        imageDimensions[0], int(imageDimensions[1]*rows/columns)))
    k = 0
    for i in range(rows):
        for j in range(columns):
            if j % nClasses == 0:
                k = int(j/nClasses)+(i*perRow)
                axarr[i, j].imshow(resizeImage.augment_image(
                    cvtColor(imread(images[k][0]), COLOR_BGR2RGB)))
                axarr[i, j].set_title(
                    (images[k][0].split(sep)[-1]).split('.')[0])
            else:
                mask = imread(labels[j % nClasses-1][k][0], IMREAD_UNCHANGED)
                axarr[i, j].imshow(resizeImage.augment_segmentation_maps(
                    [SegmentationMapsOnImage(mask, shape=mask.shape)])[0].get_arr())
                axarr[i, j].set_title(classNames[j % nClasses-1])
            axarr[i, j].axis(False)
    plt.subplots_adjust(wspace=0, hspace=0.0675,
                        left=0, right=1, bottom=0, top=1)
    return figure


# Genera una matriz de confusión a partir de un modelo y datos de validación
def confusionMatrix(model, validationImages, validationLabels, classNames, imageDimensions):
    resizeCM = resize(imageDimensions)
    images = model.predict(array([resizeCM.augment_image(cvtColor(imread(
        image[0]), COLOR_BGR2RGB)) for image in validationImages]).astype('float32')/255.0, batch_size=1)
    imagePredictions = []
    for i in range(images.shape[-1]):
        for p in [(img[..., i]*255.0).astype(uint8) for img in images]:
            _, thresh = threshold(p, 127, 255, THRESH_BINARY)
            imagePredictions.append(thresh*(i+1)/255)
    cm = confusion_matrix(([[lsi.get_arr() for lsi in ls] for ls in [resizeCM.augment_segmentation_maps([SegmentationMapsOnImage(l, shape=l.shape) for l in ls]) for ls in [
                          [imread(str(l[0]), IMREAD_UNCHANGED) for l in ls] for ls in validationLabels]]].flatten()).astype(uint8), (array(imagePredictions).flatten()).astype(uint8))
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion matrix')
    plt.colorbar()
    classNames = concatenate([['empty'], classNames])
    tickMarks = arange(len(classNames))
    plt.xticks(tickMarks, classNames, rotation=45)
    plt.yticks(tickMarks, classNames)
    cm = around(cm.astype('float') / cm.sum(axis=1)[:, newaxis], decimals=2)
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment='center',
                 color='white' if cm[i, j] > cm.max() / 2.0 else 'black')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


# Predice un conjunto de imágenes y guarda su predicción
def predict(relativePath, testData, model, classNames):
    beforeAll = datetime.now()

    predictionsDir = join(relativePath, 'Predictions')
    if not exists(predictionsDir):
        mkdir(predictionsDir)

    for cn in classNames:
        classDir = join(predictionsDir, cn)
        if not exists(classDir):
            mkdir(classDir)

    times = []
    listPredictions = []
    for i in range(len(testData)):
        image = cvtColor(imread(testData[i][0]), COLOR_BGR2RGB)

        imageResized = resize([int(model.input.shape[1]), int(
            model.input.shape[2])]).augment_image(image)

        imagePredicted = array([imageResized/255.0]).astype(float32)

        before = datetime.now()
        prediction = model.predict(imagePredicted, batch_size=1)
        after = datetime.now()
        times.append((after - before).total_seconds())

        listClassPredictions = []
        for c in range(prediction.shape[-1]):
            imagePrediction = (prediction[0][..., c]*255.0).astype(uint8)

            _, thresh = threshold(imagePrediction, 127, 255, THRESH_BINARY)
            thresh = thresh.astype(uint8)

            mask = bitwise_not(thresh)

            blankImage = zeros(
                (int(model.input.shape[1]), int(model.input.shape[2]), 3), uint8)
            blankImage[:, :] = (63, 255, 0)

            colortrans = addWeighted(imageResized, 0.5, bitwise_and(bitwise_or(merge([mask, mask, mask]), bitwise_and(
                merge([thresh, thresh, thresh]), blankImage, None), None), imageResized, None), 0.5, 1)

            fileName = join(
                predictionsDir, classNames[c], (testData[i][0].split(sep)[-1]).split('.')[0])
            imwrite(fileName+'Original.png', cvtColor(image, COLOR_RGB2BGR))
            imwrite(fileName+'Prediction.png',
                    cvtColor(colortrans, COLOR_RGB2BGR))
            imwrite(fileName+'Mask.png', thresh)
            listClassPredictions.append(fileName+'.png')

        listPredictions.append(array(listClassPredictions).reshape(
            (len(listClassPredictions), 1)))

    afterAll = datetime.now()

    return array(listPredictions).transpose(1, 0, 2), (afterAll - beforeAll).total_seconds(), times
