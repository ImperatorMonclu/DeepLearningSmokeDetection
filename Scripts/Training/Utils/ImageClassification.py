from glob import glob
from os.path import join, exists

from imgaug.augmenters import Sequential
from imgaug.augmenters.size import CenterPadToAspectRatio, Resize
from cv2 import INTER_AREA, imread, cvtColor, COLOR_BGR2RGB, imwrite, COLOR_RGB2BGR

from tensorflow.python.keras.utils.data_utils import Sequence

from numpy import hsplit, asarray, array, floor, arange, argmax, around, newaxis, float32
from numpy.random import shuffle

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from itertools import product

from datetime import datetime
from os import mkdir


# Crear el conjunto de datos con la dirección de las imágenes
def load(relativePath, labels, n=None, validationPercentage=None):

    filenameImagesListTrain = []
    for l in labels:
        filenameImagesListTrain.append(
            list(glob(join(relativePath, 'Train', l, '*'))))

    filenameImagesListValidation = []
    for l in labels:
        filenameImagesListValidation.append(
            list(glob(join(relativePath, 'Validation', l, '*'))))

    
    if n is None:

        dataTrain = []
        for i in range(len(filenameImagesListTrain)):
            for j in range(len(filenameImagesListTrain[i])):
                dataTrain.append([filenameImagesListTrain[i][j], i])

        dataValidation = []
        if validationPercentage is None:
            for i in range(len(filenameImagesListValidation)):
                for j in range(len(filenameImagesListValidation[i])):
                    dataValidation.append([filenameImagesListValidation[i][j], i])
        else:
            for i in range(len(filenameImagesListValidation)):
                for j in range(len(int(len(filenameImagesListTrain[i])*validationPercentage))):
                    dataValidation.append([filenameImagesListValidation[i][j], i])
    else:
        if validationPercentage is None:
            validationPercentage = 0.5
            
        nTrain = int(n*(1.0-validationPercentage)/len(filenameImagesListTrain))
        nValidation = int(n*validationPercentage/len(filenameImagesListValidation))

        dataTrain = []
        for i in range(nTrain):
            for j in range(len(filenameImagesListTrain)):
                dataTrain.append([filenameImagesListTrain[j][i], j])

        dataValidation = []
        for i in range(nValidation):
            for j in range(len(filenameImagesListValidation)):
                dataValidation.append([filenameImagesListValidation[j][i], j])

    shuffle(dataTrain)
    shuffle(dataValidation)

    def splitLabelsImages(data, labels):
        dataSplit = hsplit(asarray(data), array(range(1, 2)))
        images = dataSplit[0]
        labels = dataSplit[1]
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
                                   pad_mode='edge'),
            Resize(imageDimensions,
                   interpolation=INTER_AREA)
        ])


# Genera conjunto de datos para el entrenamiento
class DataGenerator(Sequence):

    def __init__(self,
                 imagePaths,
                 labels,
                 batchSize=16,
                 resize=None,
                 augmenters=None,
                 shuffle=False):
        self.imagePaths = imagePaths
        self.labels = labels
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
        images = [cvtColor(imread(str(self.imagePaths[k][0])),
                           COLOR_BGR2RGB) for k in indexes]
        labels = array([float(self.labels[k]) for k in indexes])
        if self.resize != None:
            images = self.resize.augment_images(images)
        if self.augmenters != None:
            images = self.augmenters.augment_images(images)
        images = array([img/255.0 for img in images]).astype('float32')
        return images, labels


# Genera una imagen con un grupo de imágenes
def imageGrid(images, labels, classNames=['0'], perRow=4, imageDimensions=(64, 64), nImages=16):
    rows = int(nImages/perRow)
    figure, axarr = plt.subplots(rows, perRow, figsize=(
        imageDimensions[0], int(imageDimensions[1]*rows/perRow)))
    k = 0
    for i in range(rows):
        for j in range(perRow):
            k = int(j)+(i*perRow)
            axarr[i, j].imshow(resize(imageDimensions).augment_image(
                cvtColor(imread(images[k][0]), COLOR_BGR2RGB)))
            axarr[i, j].set_title(classNames[int(argmax(labels[k]))])
            axarr[i, j].axis(False)
    plt.subplots_adjust(wspace=0, hspace=0.125,
                        left=0, right=1, bottom=0, top=1)
    return figure


# Genera una matriz de confusión a partir de un modelo y datos de validación
def confusionMatrix(model, validationImages, validationLabels, classNames, imageDimensions):
    predictions = model.predict(array([resize(imageDimensions).augment_image(cvtColor(imread(
        image[0]), COLOR_BGR2RGB)) for image in validationImages]).astype('float32')/255.0, batch_size=1)
    cm = confusion_matrix(argmax(validationLabels, axis=1),
                          argmax(predictions, axis=1))
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion matrix')
    plt.colorbar()
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

        imwrite(join(predictionsDir, classNames[argmax(
            prediction[0], axis=1)], testData[i][0]), cvtColor(image, COLOR_RGB2BGR))

        listPredictions.append(prediction[0])

    afterAll = datetime.now()

    return array(listPredictions), (afterAll - beforeAll).total_seconds(), times
